"""
train_dpo.py
DPO 训练：加载 SFT checkpoint 初始化 policy 和 reference 两份模型权重，
用偏好对比（chosen vs rejected）优化 policy，reference 全程冻结不参与梯度更新。
 
核心公式（上一轮已经推导过，这里逐行对应到代码）：
  L_DPO = -log σ( β · [ (logπ_policy(y_w|x) - logπ_ref(y_w|x))
                        - (logπ_policy(y_l|x) - logπ_ref(y_l|x)) ] )
"""

import sys
from pathlib import Path
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import copy
import time
import argparse
import math
 
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW  # type:ignore
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter  # type:ignore
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from loguru import logger
from rich import print as rprint
 
from model.model_minimind import MiniMindForCausalLM, MiniMindConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Minimind DPO训练")
 
    parser.add_argument("--processed_dir", type=str, default="./data/processed")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--runs_dir", type=str, default="./runs")
    parser.add_argument("--sft_ckpt", type=str, required=True,
                        help="SFT训练完成的checkpoint路径，policy和reference都从这里初始化")


    parser.add_argument("--epochs", type=int, default=1)
    # 官方DPO学习率是4e-8，比SFT的2e-5小了近500倍
    parser.add_argument("--lr", type=float, default=4e-8)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO温度系数，控制policy允许偏离reference多远，越大越保守")
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--effective_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500,
                        help="每多少个optimizer step跑一次验证集，SFT阶段吃过没有验证集的亏，这次必须有")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--val_ratio", type=float, default=0.02,
                        help="从训练数据里切多少比例出来做验证集")
    parser.add_argument("--keep_last_n_ckpt", type=int, default=3,
                        help="SFT阶段设成1导致后面没法回溯对比，这次至少留3个")

    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
 
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--no_timestamp", action="store_true")

    args = parser.parse_args()
    args.accumulation_steps = args.effective_batch_size // args.micro_batch_size

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.experiment_name is None:
        base_name = f"dpo_h{args.hidden_size}_l{args.num_layers}_lr{args.lr}_beta{args.beta}"
        args.experiment_name = base_name if args.no_timestamp else f"{base_name}_{timestamp}"
    elif not args.no_timestamp:
        args.experiment_name = f"{args.experiment_name}_{timestamp}"
 
    args.processed_dir = Path(args.processed_dir)
    args.checkpoint_dir = Path(args.checkpoint_dir)
    args.runs_dir = Path(args.runs_dir)
    args.sft_ckpt = Path(args.sft_ckpt)
 
    return args

# ═══════════════════════════════════════════
# 1. DPO Dataset：一次取出chosen和rejected各自的 x/y/mask，一共6个tensor
# ═══════════════════════════════════════════
class DPODataset(Dataset):
    _DTYPE_MAP = {"uint16": np.uint16, "int16": np.int16, 
                  "uint8": np.uint8, "int8": np.int8}
    def __init__(self, processed_dir: Path, meta: dict, indices: np.ndarray = None):    # type:ignore
        ids_dtype = self._DTYPE_MAP[meta["ids_dtype"]]
        mask_dtype = self._DTYPE_MAP[meta["mask_dtype"]]
        shape = (meta["count"], meta["seq_len"])
 
        self.x_chosen = np.memmap(processed_dir / meta["files"]["x_chosen"], dtype=ids_dtype, mode="r", shape=shape)
        self.y_chosen = np.memmap(processed_dir / meta["files"]["y_chosen"], dtype=ids_dtype, mode="r", shape=shape)
        self.mask_chosen = np.memmap(processed_dir / meta["files"]["mask_chosen"], dtype=mask_dtype, mode="r", shape=shape)
        self.x_rejected = np.memmap(processed_dir / meta["files"]["x_rejected"], dtype=ids_dtype, mode="r", shape=shape)
        self.y_rejected = np.memmap(processed_dir / meta["files"]["y_rejected"], dtype=ids_dtype, mode="r", shape=shape)
        self.mask_rejected = np.memmap(processed_dir / meta["files"]["mask_rejected"], dtype=mask_dtype, mode="r", shape=shape)

        # 支持传入子集索引，用于切分train/val（SFT阶段缺的这块，这次补上）
        self.indices = indices if indices is not None else np.arange(meta["count"])
    def __len__(self):
        return len(self.indices)

    def __getitem__(self,i):
        idx = self.indices[i]
        return (
            torch.tensor(self.x_chosen[idx].astype(np.int64)),
            torch.tensor(self.y_chosen[idx].astype(np.int64)),
            torch.tensor(self.mask_chosen[idx].astype(np.float32)),  # mask要参与浮点数加权求和，转float
            torch.tensor(self.x_rejected[idx].astype(np.int64)),
            torch.tensor(self.y_rejected[idx].astype(np.int64)),
            torch.tensor(self.mask_rejected[idx].astype(np.float32)),
        )

def split_train_val(count:int,val_ratio:float,seed:int):
    "切分验证集"
    rng = np.random.default_rng(seed)
    indices = rng.permutation(count)
    val_count = max(1, int(count * val_ratio))
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    return train_indices, val_indices

# ═══════════════════════════════════════════
# 2. 核心：计算"整段回复的对数概率"
# ═══════════════════════════════════════════
def compute_sequence_logprobs(
    model,x:torch.Tensor,y:torch.Tensor,mask:torch.Tensor
) -> torch.Tensor:
    '''
    DPO核心，对应公式里的 logπ(y|x)
        步骤：
        1.model(x)拿到每一个位置、每个词表位置的logits——shape(batch,seq_len,vocab_size)
        2.log_softmax把logits转成对数概率分布
        3.gather：从vocab_size维度里，只取“真实发生的下一个token y”对应的那个log概率(词表分布不需要，只关注模型给真实答案打了多少分)
        4.乘以mask:只累加assistant回复部分的log概率，prompt部分（mask=0)不计入
        5.对seq_len维度求和：把每个token的log概率加起来，得到整句话的log概率
    '''
    output = model(x)   # 这里不需要labels，DPO不需要计算CrossEntropyLoss   
    # rprint(f"output type:{type(output)}\n,output:{output}") # 后期删除，预先查看数据格式
    logits = output.logits  # shape(batch,seq_len,vocab_size)
    log_probs = F.log_softmax(logits,dim=-1)    # shape(batch,seq_len，vocab_size)

    # gather: 对每个位置，从vocab_size个候选里，精确取出真实token y对应的那个概率
    # y.unsqueeze(-1) 把 (batch, seq_len) 变成 (batch, seq_len, 1)，用来做index
    token_log_probs = torch.gather(log_probs,dim=-1,index=y.unsqueeze(-1)).squeeze(-1)  # shape(batch,seq_len)
    # 只累加mask=1的位置（assistant回复部分），prompt部分即便算出了概率也不计入

    seq_log_probs = (token_log_probs * mask).sum(dim=-1)  # (batch,)
    return seq_log_probs

# ═══════════════════════════════════════════
# 3. DPO loss：把4个logπ值组合成最终的loss
# ═══════════════════════════════════════════
def compute_dpo_loss(
    policy_chosen_logps:torch.Tensor,
    policy_rejected_logps:torch.Tensor,
    ref_chosen_logps:torch.Tensor,
    ref_rejected_logps:torch.Tensor,
    beta:float
):
    """
    对应公式：
      L_DPO = -log σ( β · [ (logπ_policy(y_w|x) - logπ_ref(y_w|x))
                            - (logπ_policy(y_l|x) - logπ_ref(y_l|x)) ] )
 
    pi_logratios  = logπ_policy(y_w|x) - logπ_policy(y_l|x)   —— policy自己对chosen/rejected的偏好程度
    ref_logratios = logπ_ref(y_w|x)    - logπ_ref(y_l|x)      —— reference（SFT模型）本来的偏好程度
    这两者相减，得到的就是"policy相对reference，多学到了多少偏好倾向"——这正是上一轮讲的"隐式奖励"的差值
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    logits = pi_logratios - ref_logratios
    loss = - F.logsigmoid(beta * logits)    # 用于loss.backend()更新policy的权重

    # 训练时非常有用的诊断指标（不是loss的一部分，只是用来监控训练是否朝着正确方向走）：
    # "隐式奖励"本身的数值，以及chosen奖励是否真的比rejected高（这个准确率应该随训练逐渐上升）
    chosen_rewards = (beta * (policy_chosen_logps - ref_chosen_logps)).detach() # chosen样本的隐式奖励（监控指标）
    rejected_rewards = (beta * (policy_rejected_logps - ref_rejected_logps)).detach()   # rejected样本的隐式奖励（监控指标）
    reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()    # batch内的chosen奖励>rejected奖励的占比（核心观测指标）

    return loss.mean(),chosen_rewards.mean(),rejected_rewards.mean(),reward_accuracy

# ═══════════════════════════════════════════
# 4. LR Scheduler（还是复用SFT那套，warmup+余弦衰减）
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
# 5. 验证：跑一遍val集，只算指标不更新参数
# ═══════════════════════════════════════════
@torch.no_grad()
def run_eval(policy,reference,val_loader,beta,device):
    policy.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for batch in val_loader:
        x_c, y_c, mask_c, x_r, y_r, mask_r = [t.to(device) for t in batch]
        policy_chosen_logps = compute_sequence_logprobs(policy, x_c, y_c, mask_c)
        policy_rejected_logps = compute_sequence_logprobs(policy, x_r, y_r, mask_r)
        ref_chosen_logps = compute_sequence_logprobs(reference, x_c, y_c, mask_c)
        ref_rejected_logps = compute_sequence_logprobs(reference, x_r, y_r, mask_r)

        loss, _, _, acc = compute_dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            beta=beta
        )
        total_loss += loss.item()
        total_acc += acc.item()
        n_batches += 1

    policy.train()
    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)

# ═══════════════════════════════════════════
# 6. 训练主流程
# ═══════════════════════════════════════════
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.runs_dir / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    logger.info(f"TensorBoard: tensorboard --logdir {args.runs_dir}")

    # ── 加载DPO数据 ──
    logger.info("加载 DPO 数据...")
    meta_path = args.processed_dir / "dpo_meta.json"
    meta = json.load(open(meta_path, "r"))

    train_indices, val_indices = split_train_val(meta["count"], args.val_ratio, args.seed)
    logger.info(f"  总样本数: {meta['count']:,}，train: {len(train_indices):,}，val: {len(val_indices):,}")

    train_dataset = DPODataset(args.processed_dir, meta, train_indices)
    val_dataset = DPODataset(args.processed_dir, meta, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.micro_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.micro_batch_size, shuffle=False, num_workers=0)

    # ── 创建模型：policy 和 reference 各自独立的一份权重 ──
    logger.info("初始化 policy 和 reference 模型...")

    def build_model():
        return MiniMindForCausalLM(config=MiniMindConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
        ))

    if not args.sft_ckpt.exists():
        raise FileNotFoundError(f"找不到 SFT checkpoint: {args.sft_ckpt}")
    sft_state = torch.load(args.sft_ckpt, map_location=device, weights_only=True)

    # policy：会被训练更新，参与反向传播
    policy = build_model()
    policy.load_state_dict(sft_state, strict=True)
    policy.to(device)  # type:ignore
    policy.train()

    # reference：从同一份SFT权重初始化，但整个训练过程中权重永远不变
    # 用 copy.deepcopy 确保两份权重在内存里是完全独立的两份tensor，
    # 而不是共享引用——如果共享引用，policy更新时reference会被一起改掉，整个DPO机制就失效了
    reference = copy.deepcopy(policy)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)  # 显式冻结：不计算梯度，节省显存和计算

    logger.info(f"policy参数量: {sum(p.numel() for p in policy.parameters()):,}")
    logger.info(f"reference参数量: {sum(p.numel() for p in reference.parameters()):,}（冻结，不参与训练）")

    optimizer = AdamW(policy.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device, enabled=use_amp)  # type:ignore

    steps_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
    total_optim_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_optim_steps * args.warmup_ratio))
    scheduler = LambdaLR(optimizer, build_lr_lambda(total_optim_steps, warmup_steps))

    logger.info(f"开始 DPO 训练: {args.experiment_name}")
    logger.info(f"  beta: {args.beta}，学习率: {args.lr}")
    logger.info("=" * 60)

    global_step = 0
    train_start = time.time()
    saved_ckpts = []

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        for step, batch in tqdm(
            enumerate(train_loader, start=1), total=len(train_loader),
            desc=f"Epoch {epoch+1}/{args.epochs}", mininterval=1.0,
        ):
            x_c, y_c, mask_c, x_r, y_r, mask_r = [t.to(device) for t in batch]

            with torch.autocast(device_type=device, enabled=use_amp):
                # 表格里的第1、2行：policy模型，chosen和rejected各前向一次，要梯度
                policy_chosen_logps = compute_sequence_logprobs(policy, x_c, y_c, mask_c)
                policy_rejected_logps = compute_sequence_logprobs(policy, x_r, y_r, mask_r)

                # 表格里的第3、4行：reference模型，同样各前向一次，但包在no_grad里，不要梯度
                with torch.no_grad():
                    ref_chosen_logps = compute_sequence_logprobs(reference, x_c, y_c, mask_c)
                    ref_rejected_logps = compute_sequence_logprobs(reference, x_r, y_r, mask_r)

                loss, chosen_reward, rejected_reward, reward_acc = compute_dpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps, args.beta,
                )
                loss_scaled = loss / args.accumulation_steps
            scaler.scale(loss_scaled).backward()
            if step % args.accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            writer.add_scalar("Loss/train_step", loss.item(), global_step)
            writer.add_scalar("Reward/chosen", chosen_reward.item(), global_step)
            writer.add_scalar("Reward/rejected", rejected_reward.item(), global_step)
            writer.add_scalar("Reward/accuracy", reward_acc.item(), global_step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)

            if step % args.log_every == 0 or step == len(train_loader):
                tqdm.write(
                    f"Epoch {epoch+1}/{args.epochs} | Step {step}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | RewardAcc: {reward_acc.item():.2%}"
                )

            # 定期跑验证集，SFT阶段缺的这一步，这次补上
            if global_step % args.eval_every == 0:
                val_loss, val_acc = run_eval(policy, reference, val_loader, args.beta, device)
                writer.add_scalar("Loss/val", val_loss, global_step)
                writer.add_scalar("Reward/val_accuracy", val_acc, global_step)
                tqdm.write(f"  [验证集] Loss: {val_loss:.4f} | RewardAcc: {val_acc:.2%}")

        # 多保留几个checkpoint，SFT阶段keep_last_n_ckpt=1导致无法回溯对比
        ckpt_path = args.checkpoint_dir / f"{args.experiment_name}_epoch{epoch+1}.pth"
        torch.save(policy.state_dict(), ckpt_path)
        logger.info(f"Checkpoint: {ckpt_path}")
        saved_ckpts.append(ckpt_path)
        if args.keep_last_n_ckpt > 0 and len(saved_ckpts) > args.keep_last_n_ckpt:
            old_ckpt = saved_ckpts.pop(0)
            if old_ckpt.exists():
                old_ckpt.unlink()

    total_time = time.time() - train_start
    logger.info(f"DPO 训练完成！总耗时: {total_time/60:.1f} 分钟")
 
    final_path = args.checkpoint_dir / f"{args.experiment_name}_final.pth"
    torch.save(policy.state_dict(), final_path)
    logger.info(f"最终模型: {final_path}")

    writer.close()

if __name__ == "__main__":
    args = parse_args()
    train(args)

