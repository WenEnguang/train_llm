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

from model.model_minimind import MiniMindForCausalLM, MiniMindConfig

