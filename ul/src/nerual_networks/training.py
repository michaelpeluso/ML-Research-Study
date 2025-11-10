import time
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Callable, Dict, List, Optional, Tuple

from utils.logger import print_t as print

def print_experiment_config(
    part_name: str,
    dataset: str,
    method: str,
    architecture: List[int],
    device: torch.device,
    optimizer_name: str,
    learning_rate: float,
    max_updates: int,
    l_threshold: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    model: nn.Module,
    **kwargs
) -> None:
    '''
    Print detailed experiment configuration at the start of each part.
    '''
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*70}")
    print(f"{part_name.upper()} - EXPERIMENT CONFIGURATION")
    print(f"{'='*70}")
    print(f"Dataset:             {dataset}")
    print(f"Task:                {method}")
    print(f"Architecture:        {architecture}")
    print(f"Device:              {device}")
    print(f"-" * 70)
    print(f"Optimizer:           {optimizer_name}")
    print(f"Learning Rate:       {learning_rate}")
    print(f"Max Updates:         {max_updates:,}")
    if l_threshold > 0: print(f"Loss Threshold:      {l_threshold:.6f}")
    else: print(f"Loss Threshold:      N/A")
    print(f"-" * 70)
    print(f"Model Parameters:    {num_params:,} total")
    print(f"Trainable Params:    {num_trainable:,}")
    print(f"-" * 70)
    print(f"Train Samples:       {len(train_loader.dataset):,}") #type:ignore
    print(f"Val Samples:         {len(val_loader.dataset):,}") #type:ignore
    print(f"Test Samples:        {len(test_loader.dataset):,}") #type:ignore
    print(f"Train Batch Size:    {train_loader.batch_size}")
    print(f"Val Batch Size:      {val_loader.batch_size}")
    print(f"Test Batch Size:     {test_loader.batch_size}")
    print(f"Train Batches:       {len(train_loader)}")
    print(f"Val Batches:         {len(val_loader)}")
    print(f"Test Batches:        {len(test_loader)}")
    
    # Print any additional kwargs
    if kwargs:
        print(f"-" * 70)
        for key, value in kwargs.items():
            print(f"{key}:".ljust(21) + f"{value}")
    
    print(f"{'='*70}\n")

def eval_loss(model: nn.Module, loader: DataLoader, loss_fn: Callable, device: torch.device, restore_train: bool = True) -> float:
    '''
    compute average loss on loader without gradients
    '''
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if out.shape[-1] == 1:
                out = out.squeeze(-1)
            loss = loss_fn(out, y)
            total_loss += loss.item()
            num_batches += 1
    if restore_train:
        model.train()
    return total_loss / num_batches

def train_to_budget(
    model: nn.Module,
    opt: Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_updates: int,
    l_threshold: float,
    loss_fn: Callable,
    device: torch.device,
    log_interval: int = 500,
    eval_interval: Optional[int] = None,
    optimizer_name: str = "unknown",
    lr_decay_rate: float = 0.95,  # Reduced from 1.0 for better convergence
    decay_every: int = 500  # Reduced from 1000 to decay more frequently
) -> Tuple[List[float], int, float, float]:
    '''
    train model with optimizer until max_updates exceeded or val_loss <= l_threshold.
    evaluates val_loss every eval_interval (default: log_interval) updates.
    logs progress messages every log_interval updates.
    '''
    if eval_interval is None:
        eval_interval = min(log_interval, max_updates)  # Ensure at least one evaluation
    
    curves: List[float] = []
    updates = 0
    steps_to_l = max_updates  # default if not reached
    start_time = time.perf_counter()

    model.train()

    while updates < max_updates:
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            if out.shape[-1] == 1:
                out = out.squeeze(-1)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            updates += 1

            if lr_decay_rate < 1.0 and updates % decay_every == 0:
                new_lr = None
                for param_group in opt.param_groups:
                    param_group['lr'] *= lr_decay_rate
                    new_lr = param_group['lr']
                if new_lr is not None:
                    print(f"[{optimizer_name}] Decayed LR to {new_lr:.6f} at update {updates}")

            if updates % eval_interval == 0 or updates >= max_updates:
                val_loss = eval_loss(model, val_loader, loss_fn, device, restore_train=True)
                curves.append(val_loss)

                if val_loss <= l_threshold and steps_to_l == max_updates:
                    steps_to_l = updates
                
                if updates % log_interval == 0 or updates >= max_updates:
                    elapsed_time = time.perf_counter() - start_time
                    progress_pct = (updates / max_updates) * 100
                    print(f"  [{optimizer_name}] Update {updates}/{max_updates} ({progress_pct:.1f}%) | Val Loss: {val_loss:.4f} | Time: {elapsed_time:.1f}s")
                
            if updates >= max_updates:
                break

        if updates >= max_updates:
            break

    wall_time = time.perf_counter() - start_time

    final_train_loss = eval_loss(model, train_loader, loss_fn, device, restore_train=False)
    
    # Ensure we have at least one validation evaluation
    if not curves:
        final_val_loss = eval_loss(model, val_loader, loss_fn, device, restore_train=False)
        curves.append(final_val_loss)

    print(f"  [{optimizer_name}] Completed in {wall_time:.2f}s | Val: {curves[-1]:.4f} | Train: {final_train_loss:.4f} | Steps to threshold: {steps_to_l}\n")

    return curves, steps_to_l, wall_time, final_train_loss