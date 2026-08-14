"""
evaluate_pretrain.py
评估预训练模型的 Perplexity
"""
import math
import torch
from tqdm import tqdm

def evaluate(model,dataloader,device,max_eval_batches=None):
    """
    在训练循环外调用：评估当前模型的 PPL
    """

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), 
                    total=min(len(dataloader), max_eval_batches or len(dataloader)),
                    desc="Evaluating")
        for batch_idx, (input_ids, labels) in pbar:
            if max_eval_batches and batch_idx >= max_eval_batches:
                break
            
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]
            
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            tokens_in_batch = batch_size * seq_len
            
            total_loss += loss.item() * tokens_in_batch
            total_tokens += tokens_in_batch
            
            running_avg = total_loss / total_tokens
            running_ppl = math.exp(running_avg)
            pbar.set_postfix(loss=f"{running_avg:.4f}", ppl=f"{running_ppl:.2f}")
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity