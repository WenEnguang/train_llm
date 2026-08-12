import os
from pathlib import Path
import torch
import argparse
import dataclasses
from dataclasses import dataclass, fields
import sys
from loguru import logger


from model.model_minimind import MiniMindConfig

# ******************** 运行期间配置模块 ************************
ROOT_DIR = Path(__file__).resolve().parent.parent
CORE_DIR = ROOT_DIR / "core"
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUT_DIR = ROOT_DIR / "out"
MODEL_DIR = ROOT_DIR / "model"
DOCS_DIR = CORE_DIR / "docs"
LOG_DIR = ROOT_DIR / "logs"
RUNS_DIR = ROOT_DIR / "runs"  # TensorBoard 日志目录


DATASET_FILES = {
    "pretrain": "pretrain_t2t_mini.jsonl",
    "sft": "sft_t2t_mini.jsonl",
    "dpo": "dpo.jsonl",
}

def configure_domestic_mirrors() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("MODELSCOPE_DOMAIN", "www.modelscope.cn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def ensure_workspace() -> None:
    configure_domestic_mirrors()
    for path in (LOG_DIR, OUT_DIR, DOCS_DIR, RUNS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR):
        path.mkdir(parents=True, exist_ok=True)

def build_config(max_seq_len: int | None = None, **overrides) -> MiniMindConfig:
    kwargs = dict(overrides)
    if max_seq_len is not None:
        kwargs["max_position_embeddings"] = max(2048, max_seq_len)
    return MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs)


def checkpoint_path(stage: str) -> Path:
    suffix = build_config().hidden_size
    return OUT_DIR / f"{stage}_{suffix}.pth"


def state_path(stage: str) -> Path:
    return OUT_DIR / f"{stage}_state.pt"

def device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ******************** 配置模块 ***************************
@dataclass
class BaseStageConfig:
    """所有训练阶段共享的字段。"""

    epochs: int = 2
    seq_len: int = 512
    lr: float = 5e-4
    effective_batch_size: int = 48
    micro_batch_size: int = 4
    checkpoint_name: str = "pretrain"
    seed: int = 42
    log_every: int = 50
    use_tensorboard: bool = True
    resume: bool = True

@dataclass
class PretrainConfig(BaseStageConfig):
    checkpoint_name: str = "pretrain"

@dataclass
class SFTConfig(BaseStageConfig):
    epochs: int = 3
    lr: float = 2e-5
    effective_batch_size: int = 32
    micro_batch_size: int = 8
    checkpoint_name: str = "sft"

@dataclass
class DPOConfig(BaseStageConfig):
    epochs: int = 1
    lr: float = 5e-7
    effective_batch_size: int = 8
    micro_batch_size: int = 2
    checkpoint_name: str = "dpo"
    log_every: int = 20
    beta: float = 0.1

def build_arg_parser(config_cls: type, description: str | None = None) -> argparse.ArgumentParser:
    """根据 dataclass 的字段自动生成 argparse 参数（含默认值）。"""
    parser = argparse.ArgumentParser(description=description)
    defaults = config_cls()
    for f in fields(config_cls):
        default = getattr(defaults, f.name)
        flag = f"--{f.name.replace('_', '-')}"
        if f.type is bool or isinstance(default, bool):
            parser.add_argument(flag, dest=f.name, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(flag, dest=f.name, type=type(default), default=default)
    return parser

def parse_config(config_cls: type, argv: list[str] | None = None, description: str | None = None):
    """解析命令行参数并返回填充好的 config 实例。"""
    parser = build_arg_parser(config_cls, description=description)
    namespace = parser.parse_args(argv)
    values = {f.name: getattr(namespace, f.name) for f in fields(config_cls)}
    return config_cls(**values)

def as_dict(config) -> dict:
    return dataclasses.asdict(config)

# *********************** 日志配置 ******************************
_CONFIGURED = False


def setup_logging(stage: str | None = None):
    """配置 loguru 的 sink，重复调用是安全的（幂等）。"""
    global _CONFIGURED
    if _CONFIGURED:
        return logger

    logger.remove()  # 移除 loguru 默认 sink，避免重复输出
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
        "<cyan>{extra[stage]}</cyan> | <level>{message}</level>",
    )
    logger.add(
        LOG_DIR / "train_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="00:00",
        retention="14 days",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {extra[stage]} | {message}",
    )
    _CONFIGURED = True
    return logger


def get_logger(stage: str):
    """返回绑定了 stage 字段的 logger，日志里会自动带上 [pretrain]/[sft]/[dpo] 标签。"""
    setup_logging()
    return logger.bind(stage=stage)
