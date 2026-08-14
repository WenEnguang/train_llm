'''
最小预训练脚本


'''
import os
import sys
import json
import time
import argparse
from pathlib import Path

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW   # type:ignore
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter   # type:ignore
from torch.optim.lr_scheduler import LambdaLR

from model.model_minimind import MiniMindForCausalLM,MiniMindConfig
from evaluate_pretrain import evaluate


# ======================================
# 0.命令行解析
# =======================================
def parse_args():
    parser = argparse.ArgumentParser(description="minimind训练")

    # 数据参数
    parser.add_argument("--processed_dir", type=str, default="./data/processed",
                        help="预处理后的 .bin 文件目录")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="模型保存目录")
    parser.add_argument("--runs_dir", type=str, default="./runs",
                        help="TensorBoard 日志目录")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--micro_batch_size", type=int, default=4,
                        help="每个 batch 的实际样本数")
    parser.add_argument("--effective_batch_size", type=int, default=48,
                        help="通过梯度累积模拟的大 batch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10,
                        help="每多少步打印一次日志")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="warmup 步数占总训练步数的比例")
    parser.add_argument("--keep_last_n_ckpt", type=int, default=3,
                        help="只保留最近 N 个 epoch checkpoint，<=0 表示全部保留")
    parser.add_argument("--max_eval_batches", type=int, default=200,
                        help="评估时最多使用多少个 batch（200 个足够得到稳定结果）")

    # 模型参数
    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    
    # 实验名称
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="实验名称，用于 TensorBoard 日志子目录,留空自动生成实验名称")
    parser.add_argument("--no_timestamp", action="store_true",
                        help="不自动追加时间戳（不建议：相同超参数多次运行会互相覆盖 log/ckpt）")

    args = parser.parse_args()

    # 计算梯度累积步数
    args.accumulation_steps = args.effective_batch_size // args.micro_batch_size

    # 自动生成实验名称
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.experiment_name is None:
        base_name = (
            f"pretrain_h{args.hidden_size}_l{args.num_layers}_"
            f"lr{args.lr}_bs{args.effective_batch_size}"
        )
        args.experiment_name = base_name if args.no_timestamp else f"{base_name}_{timestamp}"
    elif not args.no_timestamp:
        # 用户手动指定了名字，也追加时间戳保证唯一，除非显式关闭
        args.experiment_name = f"{args.experiment_name}_{timestamp}"

    # 转成 Path
    args.processed_dir = Path(args.processed_dir)
    args.checkpoint_dir = Path(args.checkpoint_dir)
    args.runs_dir = Path(args.runs_dir)
    
    return args

# ═══════════════════════════════════════════════════════════
# 1. 数据集定义
# ═══════════════════════════════════════════════════════════
class PretrainDataset(Dataset):
    """
    从 packing 后的 .bin 文件读取数据
    每个样本是 seq_len 个 token，预测下一个 token
    """
    def __init__(self,bin_path:Path,config:dict):
        # 读取元信息
        self.seq_len = config['seq_len']
        total_tokens = config["total_tokens"]
        dtype = np.uint16 if config["dtype"] == "uint16" else np.uint32

        # 内存映射（零拷贝）
        self.data = np.memmap(
            str(bin_path),
            dtype=dtype,
            mode="r",
            shape=(total_tokens,),
        )
        
        # 计算总序列数
        self.num_sequences = total_tokens // self.seq_len
    
    def __len__(self):
        return self.num_sequences

    def __getitem__(self,idx):
        # 第idx条序列在.bin中的起始位置
        start = idx * self.seq_len
        end = start + self.seq_len

        # 读取整条序列
        seq = self.data[start:end].copy()   # copy 必须，memmap 不能直接转 tensor

        # 输入：前 seq_len-1 个 token
        # 标签：后 seq_len-1 个 token（每个位置预测下一个）
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)  # (seq_len-1,)
        labels = torch.tensor(seq[1:], dtype=torch.long)      # (seq_len-1,)
        
        return input_ids, labels

# ═══════════════════════════════════════════════════════════
# 2. LR Scheduler：warmup + cosine decay
# ═══════════════════════════════════════════════════════════
def build_lr_lambda(total_steps: int, warmup_steps: int):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# ═══════════════════════════════════════════════════════════
# 3. 训练函数
# ═══════════════════════════════════════════════════════════

def train(args):
    # ── 设备 ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    print(f"设备: {device}")

    # ── 固定随机种子 ──
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── 创建目录 ──
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.runs_dir / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── TensorBoard ──
    writer = SummaryWriter(log_dir=str(run_dir))
    print(f"TensorBoard: tensorboard --logdir {args.runs_dir}")

    # 加载数据
    print(f"加载数据")
    bin_path = args.processed_dir / 'pretrain_tokens.bin'
    config_path = args.processed_dir / "pretrain_config.json"

    with open(config_path,'r') as f:
        data_config = json.load(f)

    dataset = PretrainDataset(bin_path, data_config)
    print(f"  序列数: {len(dataset):,}")
    print(f"  序列长度: {data_config['seq_len']}")

    # Dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=0,
    )
    ## 创建eval_Dataloader
    eval_dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size * 2,  # 评估时 batch 可以大一些
        shuffle=False,
        num_workers=0,
    )
    print(f"每个epoch步数：{len(dataloader)}")

    # 初始化模型
    print(f"初始化模型")

    model = MiniMindForCausalLM(
        config=MiniMindConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        )
    ).to(device)    # type:ignore

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量：{total_params}")

    # 优化器
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device, enabled=use_amp)  # type:ignore

    # 学习率：余弦退火
    steps_per_epoch = math.ceil(len(dataloader) / args.accumulation_steps)
    total_optim_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_optim_steps * args.warmup_ratio))
    scheduler = LambdaLR(optimizer, build_lr_lambda(total_optim_steps, warmup_steps))

    # ── 训练循环 ──
    print(f"\n开始训练: {args.experiment_name}")
    print(f"  Epochs: {args.epochs}")
    print(f"  有效批次: {args.effective_batch_size} (micro={args.micro_batch_size} × accum={args.accumulation_steps})")
    print(f"  学习率: {args.lr} (warmup {warmup_steps}/{total_optim_steps} 步)")
    print("=" * 60)

    global_step = 0
    train_start = time.time()
    saved_ckpts = [] # 滚动保留最近N个checkpoint

    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_losses = []
        optimizer.zero_grad()

        for step,batch in tqdm(
            enumerate(dataloader,start=1),
            total=len(dataloader),
            desc=f"训练中。。。"
        ):
            input_ids, labels = batch
            input_ids,labels = input_ids.to(device),labels.to(device)

            # 前向传播
            with torch.autocast(device_type=device,enabled=use_amp):
                outputs = model(input_ids, labels=labels)
                raw_loss = outputs["loss"]
                assert raw_loss is not None,"模型未返回 loss，请检查 labels 是否正确传入" # 如果模型在eval模式下被误用，loss会是None
                loss = raw_loss / args.accumulation_steps

            # 反向传播
            scaler.scale(loss).backward()

            # 梯度更新
            if step % args.accumulation_steps == 0 or step == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            # 记录
            batch_loss = raw_loss.item()
            epoch_losses.append(batch_loss)
            global_step += 1

            # TensorBoard
            writer.add_scalar("Loss/step", batch_loss, global_step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)

            # 打印
            if step % args.log_every == 0 or step == len(dataloader):
                elapsed = time.time() - epoch_start
                tqdm.write(
                    f"Epoch {epoch+1}/{args.epochs} | "
                    f"Step {step:4d}/{len(dataloader)} | "
                    f"Loss: {batch_loss:.4f} | "
                    f"{elapsed:.1f}s"
                )
        # Epoch 结束
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_time = time.time() - epoch_start
        total_time = time.time() - train_start
        
        print("-" * 60)
        print(f"Epoch {epoch+1} 完成 | Avg Loss: {avg_loss:.4f} | 本轮: {epoch_time:.1f}s | 累计: {total_time:.1f}s")
        print()
        
        writer.add_scalar("Loss/epoch_avg", avg_loss, epoch + 1)
        
        # 保存 checkpoint
        ckpt_path = args.checkpoint_dir / f"{args.experiment_name}_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint: {ckpt_path}\n")

        saved_ckpts.append(ckpt_path)
        if args.keep_last_n_ckpt > 0 and len(saved_ckpts) > args.keep_last_n_ckpt:
            old_ckpt = saved_ckpts.pop(0)
            if old_ckpt.exists():
                old_ckpt.unlink()
                print(f"已删除旧 checkpoint: {old_ckpt}")
    
    total_time = time.time() - train_start
    print("=" * 60)
    print(f"训练完成！总耗时: {total_time:.1f}s ({total_time/60:.1f} 分钟)")
    
    # 保存最终模型
    final_path = args.checkpoint_dir / f"{args.experiment_name}_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"最终模型: {final_path}")


    # 评估最终模型
    final_avg_loss, final_ppl = evaluate(
        model=model, 
        dataloader=eval_dataloader, 
        device=device,
        max_eval_batches=args.max_eval_batches  # 新加的参数
    )
    print(f"模型的评估结果\nLoss:{final_avg_loss:.4f},PPL:{final_ppl:.4f}")

    # TensorBoard 记录
    writer.add_scalar("Eval/loss", final_avg_loss, args.epochs)
    writer.add_scalar("Eval/perplexity", final_ppl, args.epochs)

    # 保存评估结果到文件
    eval_result_path = args.checkpoint_dir / f"{args.experiment_name}_eval.json"
    eval_result = {
        "experiment_name": args.experiment_name,
        "checkpoint": str(final_path),
        "avg_loss": final_avg_loss,
        "perplexity": final_ppl,
        "eval_tokens": len(dataset) * data_config["seq_len"],
    }
    with open(eval_result_path, "w") as f:
        json.dump(eval_result, f, indent=2)
    print(f"评估结果已保存: {eval_result_path}")
    
    writer.close()



if __name__ == "__main__":
    args = parse_args()

    train(args)

