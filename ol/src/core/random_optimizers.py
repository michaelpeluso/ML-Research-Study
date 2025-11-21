'''
Random Optimization Algorithms for Neural Network Parameter Tuning
Implements three common random optimization strategies:

- RHC (Randomized Hill Climbing):
    Iterative local search with multiple random restarts.
    Adjusts Gaussian perturbations to escape local optima.

- SA (Simulated Annealing):
    Probabilistic search accepting worse solutions based on a temperature schedule.
    Gradually reduces exploration to refine convergence.

- GA (Genetic Algorithm):
    Population-based evolutionary search with tournament selection, single-point crossover,
    and Gaussian mutation to maintain diversity.
'''

import time
import torch
import torch.nn as nn
from typing import Callable, List, Tuple

from utils.logger import MLLogger

# flatten trainable params into 1d tensor for ro search
def get_trainable_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad])

# unflatten and set params back to model
def set_trainable_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    offset = 0
    for p in model.parameters():
        if p.requires_grad:
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset + numel].view_as(p))
            offset += numel

# progress logging for random optimization algorithms
def log_ro_progress(
    algo_name: str,
    evals: int,
    max_evals: int,
    best_loss: float,
    training_start: float,
    extra_info: str = ""
) -> None:
    """Log progress for random optimization algorithms every N evaluations"""
    elapsed_time = time.perf_counter() - training_start
    progress_pct = (evals / max_evals) * 100
    extra = f" | {extra_info}" if extra_info else ""
    print(f"  [{algo_name}] Eval {evals}/{max_evals} ({progress_pct:.1f}%) | Best Loss: {best_loss:.4f}{extra} | Time: {elapsed_time:.1f}s")

def log_ga_generation(
    generation: int,
    evals: int,
    max_evals: int,
    best_fitness: float,
    training_start: float
) -> None:
    """Log progress for genetic algorithm per generation"""
    elapsed_time = time.perf_counter() - training_start
    progress_pct = (evals / max_evals) * 100
    print(f"  [GA] Generation {generation}: evals={evals}/{max_evals} ({progress_pct:.1f}%) | Best Fitness: {best_fitness:.4f} | Time: {elapsed_time:.1f}s")

# compute val loss as ro objective, counts as 1 func eval
def validation_objective(
    flat_params: torch.Tensor,
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    device: torch.device,
    val_subset_batches: int | None = None,  # if set, only use this many batches from val_loader
) -> float:
    set_trainable_params(model, flat_params)
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        # iterate over validation loader but optionally limit to a small number of batches
        for batch_idx, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            if out.shape[-1] == 1:
                out = out.squeeze(-1)
            loss = loss_fn(out, y)
            total_loss += loss.item()
            num_batches += 1
            if val_subset_batches is not None and batch_idx + 1 >= val_subset_batches:
                break
    # protect against zero-division
    return total_loss / num_batches if num_batches > 0 else float('inf')


"""
Randomized Hill Climbing (RHC) for Random Optimization.
Design: restarts=5, gaussian perturbation ~N(0, scale), exponential decay_rate=0.99.
Budget: stops at max_evals func evals.
Returns: optimized model, history [(evals, loss)] for analysis.
Optional: uniform plateau rule for early stopping if improvement < min_delta.
"""
def rhc(
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        device: torch.device,
        restarts: int = 5,
        max_evals: int = 10000,
        initial_perturb_scale: float = 0.1,
        decay_rate: float = 0.995,
        plateau_threshold: int = 250,  # max plateauing evals
        min_delta: float = 1e-6,  # min improvement threshold
        val_subset_batches: int | None = None,
        logger: MLLogger | None = None
    ) -> Tuple[nn.Module, List[Tuple[int, float]]]:

    print(f"[RHC] Starting with {sum(p.numel() for p in model.parameters())} parameters, restarts={restarts}, max_evals={max_evals}")
    if logger is None: logger = MLLogger()
    
    last_improvement_eval = 1  # plateau tracking
    best_flat = get_trainable_params(model)
    
    # Measure single evaluation time
    eval_start = time.perf_counter()
    best_loss = initial_loss = validation_objective(best_flat, model, val_loader, loss_fn, device, val_subset_batches)
    single_eval_time = time.perf_counter() - eval_start
    
    history = [(1, initial_loss)]  # initial eval
    evals = 1

    with logger.log_step("Randomized Hill Climbing") as step_info:
        training_start = time.perf_counter()  # Start timing the training loop
        step_info.update({
            'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),  # trainable params
            'restarts': restarts,
            'initial_loss': initial_loss,
            'max_evals': max_evals,
            'initial_perturb_scale': initial_perturb_scale,
            'decay_rate': decay_rate,
            'plateau_threshold': plateau_threshold,
            'min_delta': min_delta,
        })

        restart_losses = []  # track losses per restart
        for r in range(restarts):
            perturb_scale = initial_perturb_scale  # reset perturb_scale for each restart
            print(f"[RHC] Restart {r+1}/{restarts} started with perturb_scale={perturb_scale:.6f}")
            last_improvement_eval = evals  # reset plateau tracker per restart
            # random restart with scaled gaussian noise
            current_flat = best_flat + torch.randn_like(best_flat, device=device) * perturb_scale * (r + 1)
            current_loss = validation_objective(current_flat, model, val_loader, loss_fn, device, val_subset_batches)
            improvement = best_loss - current_loss
            if improvement > min_delta:
                last_improvement_eval = evals
                best_flat = current_flat.clone()
                best_loss = current_loss
                print(f"[RHC] Initial point at eval {evals} better than best: loss={current_loss:.6f} (improvement={improvement:.6f})")
            evals += 1
            history.append((evals, current_loss))
            restart_losses.append((evals, current_loss))

            while evals < max_evals:
                # perturb with gaussian noise and decay scale
                perturb = torch.randn_like(best_flat, device=device) * perturb_scale
                current_flat = best_flat + perturb
                current_loss = validation_objective(current_flat, model, val_loader, loss_fn, device, val_subset_batches)
                improvement = best_loss - current_loss
                if improvement > min_delta:
                    best_flat = current_flat.clone()
                    best_loss = current_loss
                    last_improvement_eval = evals
                perturb_scale *= decay_rate
                evals += 1
                history.append((evals, current_loss))
                
                # Progress logging every 500 evals
                if evals % 500 == 0:
                    log_ro_progress("RHC", evals, max_evals, best_loss, training_start)

                if evals - last_improvement_eval > plateau_threshold:
                    print(f"[RHC] Stopping early at eval {evals} due to plateau")
                    break

            # log restart-specific data
            logger.log_metric(f'restart_{r+1}_best_loss', min(l[1] for l in history[-1:] if l[0] >= restart_losses[r][0]))
            logger.log_metric(f'restart_{r+1}_evals', evals - restart_losses[r][0])

        # set best individual to model
        set_trainable_params(model, best_flat)
        training_time = time.perf_counter() - training_start
        print(f"[RHC] Completed: total_evals={evals}, final_best_loss={best_loss:.6f}, training_time={training_time:.3f}s")
        
        step_info.update({
            'best_loss': best_loss,
            'evals': evals,
            'single_eval_duration': single_eval_time,
            'training_duration': training_time,
            'history': history,
            'restart_losses': restart_losses  # log all restart losses
        })
    return model, history

"""
Simulated Annealing (SA) for Random Optimization.
Design: temperature schedule with cooling, accepts worse solutions probabilistically.
Budget: stops at max_evals func evals.
Returns: optimized model, history [(evals, loss)] for analysis.
"""
def sa(
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        device: torch.device,
        max_evals: int = 10000,
        initial_temp: float = 0.1,
        min_temp: float = 0.001,
        cooling_rate: float = 0.003,
        initial_perturb_scale: float = 0.1,
        perturb_decay: float = 0.995,
        plateau_threshold: int = 1000,
        min_delta: float = 1e-6,
        val_subset_batches: int | None = None,
        logger: MLLogger | None = None
    ) -> Tuple[nn.Module, List[Tuple[int, float]]]:
    print(f"[SA] Starting with {sum(p.numel() for p in model.parameters())} parameters, max_evals={max_evals}")
    if logger is None: logger = MLLogger()
    
    best_flat = get_trainable_params(model)
    
    # Measure single evaluation time
    eval_start = time.perf_counter()
    best_loss = validation_objective(best_flat, model, val_loader, loss_fn, device, val_subset_batches)
    single_eval_time = time.perf_counter() - eval_start
    
    # SA needs to track both current state and best state
    current_flat = best_flat.clone()
    current_loss = best_loss
    
    history = [(1, best_loss)]
    evals = 1
    temp = initial_temp
    perturb_scale = initial_perturb_scale

    with logger.log_step("Simulated Annealing") as step_info:
        training_start = time.perf_counter()  # Start timing the training loop
        step_info.update({
            'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'max_evals': max_evals,
            'initial_temp': initial_temp,
            'min_temp': min_temp,
            'cooling_rate': cooling_rate,
            'initial_perturb_scale': initial_perturb_scale,
            'perturb_decay': perturb_decay,
            'plateau_threshold': plateau_threshold,
            'min_delta': min_delta,
            'initial_loss': best_loss
        })

        last_improvement_eval = 1
        while evals < max_evals and temp > min_temp:
            # Perturb from current state with decaying scale
            candidate_flat = current_flat + torch.randn_like(current_flat, device=device) * perturb_scale
            candidate_loss = validation_objective(candidate_flat, model, val_loader, loss_fn, device, val_subset_batches)
            improvement = current_loss - candidate_loss
            
            import math
            # Accept if better OR with probability exp(improvement/temp)
            # When improvement < 0 (worse), acceptance probability = exp(negative_value/temp)
            accept = False
            if improvement > 0:
                accept = True
            elif temp > 0:
                acceptance_prob = math.exp(improvement / temp)  # improvement is negative
                accept = torch.rand(1, device=device).item() < acceptance_prob
            
            if accept:
                current_flat = candidate_flat.clone()
                current_loss = candidate_loss
                
                # Update best if this is actually better
                if candidate_loss < best_loss - min_delta:
                    best_flat = candidate_flat.clone()
                    best_loss = candidate_loss
                    last_improvement_eval = evals
            
            evals += 1
            history.append((evals, best_loss))  # Log BEST loss for monotonic curve
            temp *= (1 - cooling_rate)
            perturb_scale *= perturb_decay  # Decay perturbation scale
            
            # Progress logging every 500 evals
            if evals % 500 == 0:
                log_ro_progress("SA", evals, max_evals, best_loss, training_start, extra_info=f"Temp: {temp:.4f}, Scale: {perturb_scale:.4f}")

            if evals - last_improvement_eval > plateau_threshold:
                print(f"[SA] Stopping early at eval {evals} due to plateau")
                break

        set_trainable_params(model, best_flat)
        training_time = time.perf_counter() - training_start
        print(f"[SA] Completed: total_evals={evals}, final_best_loss={best_loss:.6f}, training_time={training_time:.3f}s")
        
        step_info.update({
            'best_loss': best_loss,
            'evals': evals,
            'single_eval_duration': single_eval_time,
            'training_duration': training_time,
            'history': history,
            'final_temp': temp,
            'final_perturb_scale': perturb_scale
        })
    return model, history

"""
Genetic Algorithm (GA) for Random Optimization.
Design: population-based with tournament selection, crossover, mutation.
Budget: stops at max_evals func evals.
Returns: optimized model, history [(evals, loss)] for analysis.
"""
def ga(
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        device: torch.device,
        max_evals: int = 10000,
        pop_size: int = 50,
        mutation_rate: float = 0.1,
        mutation_std: float = 0.01,
        crossover_rate: float = 0.7,
        plateau_threshold: int = 1000,
        min_delta: float = 1e-6,
        val_subset_batches: int | None = None,
        logger: MLLogger | None = None
    ) -> Tuple[nn.Module, List[Tuple[int, float]]]:
    print(f"[GA] Starting with {sum(p.numel() for p in model.parameters())} parameters, pop_size={pop_size}, max_evals={max_evals}")
    if logger is None: logger = MLLogger()
    
    pop = [get_trainable_params(model) + torch.randn_like(get_trainable_params(model), device=device) * 0.1 for _ in range(pop_size)]
    
    # Measure single evaluation time (average of initial population)
    eval_start = time.perf_counter()
    fitness = [validation_objective(ind, model, val_loader, loss_fn, device, val_subset_batches) for ind in pop]
    single_eval_time = (time.perf_counter() - eval_start) / len(pop)
    
    evals = pop_size
    history = [(evals, min(fitness))]
    generation = 1

    with logger.log_step("Genetic Algorithm") as step_info:
        training_start = time.perf_counter()  # Start timing the training loop
        step_info.update({
            'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'max_evals': max_evals,
            'pop_size': pop_size,
            'mutation_rate': mutation_rate,
            'mutation_std': mutation_std,
            'crossover_rate': crossover_rate,
            'plateau_threshold': plateau_threshold,
            'min_delta': min_delta,
            'initial_fitness': min(fitness),
            'selection_method': 'top-k (elitist, top 50%)',
            'crossover_method': 'single-point at random position',
            'mutation_method': 'gaussian noise N(0, mutation_std)',
            'elitism': True
        })

        last_improvement_eval = evals
        prev_min_fit = min(fitness)
        while evals < max_evals:
            # selection
            selected_indices = torch.topk(torch.tensor(fitness, device=device), k=pop_size//2, largest=False).indices
            parents = [pop[i] for i in selected_indices]

            # crossover and mutation
            next_pop = [pop[torch.argmin(torch.tensor(fitness, device=device))]]  # elite
            while len(next_pop) < pop_size:
                idx1, idx2 = torch.randint(0, len(parents), (2,), device=device).tolist()
                p1, p2 = parents[idx1], parents[idx2]
                if torch.rand(1, device=device).item() < crossover_rate:
                    cross_pt = int(torch.rand(1, device=device).item() * len(p1))
                    o1 = torch.cat((p1[:cross_pt], p2[cross_pt:]))
                    o2 = torch.cat((p2[:cross_pt], p1[cross_pt:]))
                else:
                    o1, o2 = p1.clone(), p2.clone()
                if torch.rand(1, device=device).item() < mutation_rate:
                    o1 += torch.randn_like(o1, device=device) * mutation_std
                if torch.rand(1, device=device).item() < mutation_rate:
                    o2 += torch.randn_like(o2, device=device) * mutation_std
                next_pop.extend([o1, o2])

            # evaluate new population
            new_fitness = [validation_objective(ind, model, val_loader, loss_fn, device, val_subset_batches) for ind in next_pop[1:]]
            evals += len(next_pop) - 1
            fitness = [fitness[0]] + new_fitness
            pop = next_pop

            min_fit = min(fitness)
            history.append((evals, min_fit))
            
            # Progress logging every generation
            log_ga_generation(generation, evals, max_evals, min_fit, training_start)

            # log generation data
            logger.log_metric(f'generation_{generation}_best_fitness', min_fit)
            logger.log_metric(f'generation_{generation}_evals', evals - (pop_size * generation))

            improvement = prev_min_fit - min_fit
            if improvement > min_delta:
                last_improvement_eval = evals
            else:
                if evals - last_improvement_eval > plateau_threshold:
                    print(f"[GA] Stopping early at eval {evals} due to plateau")
                    break
            prev_min_fit = min_fit

            if evals >= max_evals:
                break
            generation += 1

        # set best individual to model
        best_idx = int(torch.argmin(torch.tensor(fitness, device=device)).item())
        set_trainable_params(model, pop[best_idx])
        training_time = time.perf_counter() - training_start
        print(f"[GA] Completed: total_evals={evals}, final_best_loss={min(fitness):.6f}, training_time={training_time:.3f}s")
        
        step_info.update({
            'best_loss': min(fitness),
            'evals': evals,
            'single_eval_duration': single_eval_time,
            'training_duration': training_time,
            'history': history,
            'generation_count': generation
        })
    return model, history