"""
train_sft.py
SFT 训练：加载 pretrain checkpoint，微调对话能力
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import argparse
import math

import numpy as np
import torch    
from torch.optim import AdamW   # type:ignore
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter   # type:ignore
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from loguru import logger

from model.model_minimind import MiniMindForCausalLM, MiniMindConfig


# ═══════════════════════════════════════════
# 0. 命令行解析
# ═══════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Minimind SFT训练")
    
    # 路径参数
    parser.add_argument("--processed_dir", type=str, default="./data/processed")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--runs_dir", type=str, default="./runs")
    parser.add_argument("--pretrain_ckpt", type=str, required=True,
                        help="预训练 checkpoint 完整路径")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--effective_batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--keep_last_n_ckpt", type=int, default=1)
    
    # 模型参数（必须和 pretrain 一致）
    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    
    # 实验名称
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--no_timestamp", action="store_true")
    
    args = parser.parse_args()
    
    # 计算梯度累积步数
    args.accumulation_steps = args.effective_batch_size // args.micro_batch_size
    
    # 自动生成实验名称
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.experiment_name is None:
        base_name = f"sft_h{args.hidden_size}_l{args.num_layers}_lr{args.lr}_bs{args.effective_batch_size}"
        args.experiment_name = base_name if args.no_timestamp else f"{base_name}_{timestamp}"
    elif not args.no_timestamp:
        args.experiment_name = f"{args.experiment_name}_{timestamp}"
    
    # 转成 Path
    args.processed_dir = Path(args.processed_dir)
    args.checkpoint_dir = Path(args.checkpoint_dir)
    args.runs_dir = Path(args.runs_dir)
    args.pretrain_ckpt = Path(args.pretrain_ckpt)   
    
    return args


# ═══════════════════════════════════════════
# 1. SFT Dataset
# ═══════════════════════════════════════════

class SFTDataset(Dataset):
    """读取 SFT 预处理后的二维 .bin 文件，dtype 从 meta 动态读取，避免读写不一致"""

    _NUMPY_DTYPE_MAP = {
        "int16": np.int16, "uint16": np.uint16,
        "int32": np.int32, "uint32": np.uint32,
    }

    def __init__(self, input_bin: Path, label_bin: Path, count: int, seq_len: int,
                 input_ids_dtype: str, labels_dtype: str):
        input_np_dtype = self._NUMPY_DTYPE_MAP[input_ids_dtype]
        labels_np_dtype = self._NUMPY_DTYPE_MAP[labels_dtype]

        self.input_ids = np.memmap(
            str(input_bin), dtype=input_np_dtype, mode="r", shape=(count, seq_len),
        )
        self.labels = np.memmap(
            str(label_bin), dtype=labels_np_dtype, mode="r", shape=(count, seq_len),
        )

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_ids[idx].astype(np.int64)),  # ✅ uint16→int64需显式astype，直接tensor转换对uint16支持有限
            torch.tensor(self.labels[idx].astype(np.int64)),
        )


# ═══════════════════════════════════════════
# 2. LR Scheduler
# ═══════════════════════════════════════════

def build_lr_lambda(total_steps: int, warmup_steps: int):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# ═══════════════════════════════════════════
# 3. 训练函数
# ═══════════════════════════════════════════

def train(args):
    # ── 设备 ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    
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
    logger.info(f"TensorBoard: tensorboard --logdir {args.runs_dir}")
    
    # ── 加载 SFT 数据 ──
    logger.info("加载 SFT 数据...")
    meta_path = args.processed_dir / "sft_meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    dataset = SFTDataset(
        input_bin=args.processed_dir / meta["files"]["input_ids"],
        label_bin=args.processed_dir / meta["files"]["labels"],
        count=meta["count"],
        seq_len=meta["seq_len"],
        input_ids_dtype=meta["input_ids_dtype"],   
        labels_dtype=meta["labels_dtype"],          
    )
    logger.info(f"  对话数: {len(dataset):,}")
    logger.info(f"  序列长度: {meta['seq_len']}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=0,
    )
    logger.info(f"  每个 epoch 步数: {len(dataloader):,}")
    
    # ── 创建模型 ──
    logger.info("初始化模型...")
    model = MiniMindForCausalLM(config=MiniMindConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
    )).to(device)   # type:ignore
    
    # ── 加载 pretrain checkpoint ──
    if not args.pretrain_ckpt.exists():
        raise FileNotFoundError(f"找不到 pretrain checkpoint: {args.pretrain_ckpt}")
    
    logger.info(f"加载 pretrain checkpoint: {args.pretrain_ckpt}")
    pretrain_state = torch.load(args.pretrain_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(pretrain_state, strict=True)
    logger.info("✅ 已加载 pretrain 权重")
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {total_params:,}")
    
    model.train()
    
    # ── 优化器 ──
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device, enabled=use_amp)  # type:ignore
    
    # ── LR Scheduler ──
    steps_per_epoch = math.ceil(len(dataloader) / args.accumulation_steps)
    total_optim_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_optim_steps * args.warmup_ratio))
    scheduler = LambdaLR(optimizer, build_lr_lambda(total_optim_steps, warmup_steps))
    
    # ── 训练循环 ──
    logger.info(f"开始 SFT 训练: {args.experiment_name}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  有效批次: {args.effective_batch_size} (micro={args.micro_batch_size} × accum={args.accumulation_steps})")
    logger.info(f"  学习率: {args.lr} (warmup {warmup_steps}/{total_optim_steps} 步)")
    logger.info("=" * 60)
    
    global_step = 0
    train_start = time.time()
    saved_ckpts = []
    is_tty = sys.stdout.isatty()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_losses = []
        optimizer.zero_grad()
        
        for step, batch in tqdm(
            enumerate(dataloader, start=1),
            total=len(dataloader),
            desc=f"Epoch {epoch+1}/{args.epochs}",
            disable=not is_tty, # 非终端环境下彻底禁用进度条动画
            mininterval=1.0,           # 即便是终端，也把默认0.1s刷新间隔放宽到1s，减少输出量
        ):
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # ✅ 数据验证
            assert labels.max().item() < args.vocab_size, \
                f"labels 中有超出词表的 id: {labels.max().item()} > {args.vocab_size}"
            
            # 前向传播
            with torch.autocast(device_type=device, enabled=use_amp):
                outputs = model(input_ids, labels=labels)   # ✅ 关键字参数
                raw_loss = outputs.loss                     # ✅ 属性访问
                assert raw_loss is not None, "模型未返回 loss"
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
                    f"Step {step}/{len(dataloader)} | "
                    f"Loss: {batch_loss:.4f} | "
                    f"{elapsed:.1f}s"
                )
        
        # Epoch 结束
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_time = time.time() - epoch_start
        total_time = time.time() - train_start
        
        logger.info("-" * 60)
        logger.info(f"Epoch {epoch+1} 完成 | Avg Loss: {avg_loss:.4f} | 本轮: {epoch_time:.1f}s | 累计: {total_time:.1f}s")
        
        writer.add_scalar("Loss/epoch_avg", avg_loss, epoch + 1)
        
        # 保存 checkpoint
        ckpt_path = args.checkpoint_dir / f"{args.experiment_name}_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"Checkpoint: {ckpt_path}")
        
        saved_ckpts.append(ckpt_path)
        if args.keep_last_n_ckpt > 0 and len(saved_ckpts) > args.keep_last_n_ckpt:
            old_ckpt = saved_ckpts.pop(0)
            if old_ckpt.exists():
                old_ckpt.unlink()
                logger.info(f"已删除旧 checkpoint: {old_ckpt}")
    
    # ── 训练完成 ──
    total_time = time.time() - train_start
    logger.info("=" * 60)
    logger.info(f"SFT 训练完成！总耗时: {total_time:.1f}s ({total_time/60:.1f} 分钟)")
    
    final_path = args.checkpoint_dir / f"{args.experiment_name}_final.pth"
    torch.save(model.state_dict(), final_path)
    logger.info(f"最终模型: {final_path}")
    
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)