import time
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Callable, Dict, List, Tuple

def eval_loss(model: nn.Module, loader: DataLoader, loss_fn: Callable, device: torch.device) -> float:
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
    log_interval: int = 100,
    optimizer_name: str = "unknown",  # name of the optimizer for tagging logs
    lr_decay_rate: float = 1.0,  # optional decay factor (e.g., 0.995 for exp decay)
    decay_every: int = 1000  # apply decay every N updates
) -> Tuple[List[float], List[Dict[str, float]], int, float, float]:
    '''
    train model with optimizer until max_updates exceeded or val_loss <= l_threshold.
    logs val_loss curves and grad_norms every log_interval updates.
    '''
    curves: List[float] = []
    grad_norms: List[Dict[str, float]] = []
    updates = 0
    steps_to_l = max_updates  # default if not reached
    start_time = time.perf_counter()

    model.train()

    # log start of training with device and parameters
    print(f"[{optimizer_name}] Starting training on {device} with max_updates={max_updates}, l_threshold={l_threshold:.4f}, log_interval={log_interval}")

    while updates < max_updates:
        epoch_start_time = time.perf_counter()  # track time per epoch
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
                    new_lr = param_group['lr']  # capture last for logging
                if new_lr is not None:
                    print(f"[{optimizer_name}] Decayed LR to {new_lr:.6f} at update {updates}")

            if updates % log_interval == 0:
                val_loss = eval_loss(model, val_loader, loss_fn, device)
                curves.append(val_loss)

                norms = {}
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        norms[name] = p.grad.norm().item()
                grad_norms.append(norms)

                if val_loss <= l_threshold and steps_to_l == max_updates:
                    steps_to_l = updates
                
            if updates >= max_updates:
                break

        # log epoch completion
        epoch_time = time.perf_counter() - epoch_start_time
        if curves: print(f"[{optimizer_name}] Completed epoch, time={epoch_time:.2f}s, updates={updates}, val_loss={curves[-1]:.4f}")
        else: print(f"[{optimizer_name}] Completed epoch, time={epoch_time:.2f}s, updates={updates}, val_loss=N/A")

        if updates >= max_updates:
            break

    wall_time = time.perf_counter() - start_time

    # compute final train loss
    final_train_loss = eval_loss(model, train_loader, loss_fn, device)

    # log training completion with final metrics
    print(f"[{optimizer_name}] Training completed: steps_to_l={steps_to_l}, wall_time={wall_time:.2f}s, final_val_loss={curves[-1]:.4f}, final_train_loss={final_train_loss:.4f}")

    return curves, grad_norms, steps_to_l, wall_time, final_train_loss