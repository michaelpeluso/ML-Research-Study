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

from src.utils.logger import MLLogger

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

# compute val loss as ro objective, counts as 1 func eval
def validation_objective(flat_params: torch.Tensor, model: nn.Module, val_loader: torch.utils.data.DataLoader, loss_fn: Callable, device: torch.device) -> float:
    set_trainable_params(model, flat_params)
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches


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
        plateau_threshold: int = 1000,  # max plateauing evals
        min_delta: float = 1e-6,  # min improvement threshold
        logger: MLLogger | None = None
    ) -> Tuple[nn.Module, List[Tuple[int, float]]]:

    print(f"[RHC] Starting with {sum(p.numel() for p in model.parameters())} parameters, restarts={restarts}, max_evals={max_evals}")
    if logger is None: logger = MLLogger()
    start_time = time.perf_counter()
    last_improvement_eval = 1  # plateau tracking
    best_flat = get_trainable_params(model)
    best_loss = validation_objective(best_flat, model, val_loader, loss_fn, device)
    history = [(1, best_loss)]  # initial eval
    evals = 1

    with logger.log_step("Randomized Hill Climbing") as step_info:

        for r in range(restarts):
            perturb_scale = initial_perturb_scale  # reset perturb_scale for each restart
            print(f"[RHC] Restart {r+1}/{restarts} started with perturb_scale={perturb_scale:.6f}")
            last_improvement_eval = evals  # reset plateau tracker per restart
            # random restart with scaled gaussian noise
            current_flat = best_flat + torch.randn_like(best_flat, device=device) * perturb_scale * (r + 1)
            current_loss = validation_objective(current_flat, model, val_loader, loss_fn, device)
            improvement = best_loss - current_loss  # compare initial point to global best
            if improvement > min_delta:
                last_improvement_eval = evals  # reset if initial point is better
                print(f"[RHC] Initial point at eval {evals} better than best: loss={current_loss:.6f}")
            evals += 1
            history.append((evals, current_loss))

            while evals < max_evals:
                # perturb with gaussian noise and decay scale
                perturb = torch.randn_like(current_flat, device=device) * perturb_scale
                candidate_flat = current_flat + perturb
                cand_loss = validation_objective(candidate_flat, model, val_loader, loss_fn, device)
                evals += 1
                history.append((evals, current_loss))  # log current best

                # accept if significant improvement
                improvement = current_loss - cand_loss  # positive if better
                if improvement > min_delta:
                    current_flat = candidate_flat
                    current_loss = cand_loss
                    last_improvement_eval = evals  # reset plateau tracker
                    print(f"[RHC] Improvement at eval {evals}/{max_evals}: loss improved to {cand_loss:.6f}")
                else:
                    # check for plateau
                    if evals - last_improvement_eval > plateau_threshold:
                        print(f"[RHC] Stopping early at eval {evals} due to plateau (improvement < {min_delta}) for {plateau_threshold}+ evals")
                        break

                # decay step size exponentially
                perturb_scale *= decay_rate

                if evals % 100 == 0:  # progress log every 100 evals
                    print(f"[RHC] Progress in restart {r+1}: eval {evals}/{max_evals}, current_loss={current_loss:.6f}, scale={perturb_scale:.6f}")
                if evals >= max_evals:
                    break

            # update global best if improved
            if current_loss < best_loss:
                best_flat = current_flat
                best_loss = current_loss
                print(f"[RHC] Global best updated after restart {r+1}: best_loss={best_loss:.6f}")

        # set final best params to model
        set_trainable_params(model, best_flat)
        print(f"[RHC] Completed: total_evals={evals}, final_best_loss={best_loss:.6f}")

        step_info.update({
            'parameters': sum(p.numel() for p in model.parameters()),
            'max_evals': max_evals,
            'device': str(device),
            'restarts': restarts,
            'initial_perturb_scale': initial_perturb_scale,
            'decay_rate': decay_rate,
            'plateau_threshold': plateau_threshold,
            'min_delta': min_delta,
            'perturbation_distribution': 'gaussian N(0, scale)',
            'evals': evals,
            'best_loss': best_loss,
            'wall_clock_time': time.perf_counter() - start_time,
            'history': history
        })
        return model, history


"""
Simulated Annealing (SA) for Random Optimization.
Design: initial_temp=10.0, linear decay temp / (i+1), gaussian perturbation with fixed step_size=0.1.
Budget: stops at max_evals func evals.
Returns: optimized model, history [(evals, loss)] for analysis.
Optional: uniform plateau rule for early stopping if improvement < min_delta.
"""
def sa(
        model: nn.Module, 
        val_loader: torch.utils.data.DataLoader, 
        loss_fn: Callable, 
        device: torch.device, 
        max_evals: int = 10000, 
        initial_temp: float = 10.0, 
        step_size: float = 0.1, 
        plateau_threshold: int = 1000, 
        min_delta: float = 1e-6,  # min improvement threshold
        logger: MLLogger | None = None
    ) -> Tuple[nn.Module, List[Tuple[int, float]]]:
    print(f"[SA] Starting with {sum(p.numel() for p in model.parameters())} parameters, max_evals={max_evals}, initial_temp={initial_temp}")
    if logger is None: logger = MLLogger()
    start_time = time.perf_counter()
    last_improvement_eval = 1  # plateau tracking
    best_flat = get_trainable_params(model)
    best_loss = validation_objective(best_flat, model, val_loader, loss_fn, device)
    current_flat = best_flat.clone()
    current_loss = best_loss
    evals = 1
    history = [(evals, best_loss)]

    with logger.log_step("Simulated Annealing") as step_info:

        i = 0
        while evals < max_evals:
            # perturb with gaussian noise
            candidate_flat = current_flat + torch.randn_like(current_flat, device=device) * step_size
            cand_loss = validation_objective(candidate_flat, model, val_loader, loss_fn, device)
            evals += 1
            history.append((evals, best_loss))

            # update best if significant improvement
            improvement = best_loss - cand_loss  # positive if better
            if improvement > min_delta:
                best_flat = candidate_flat.clone()
                best_loss = cand_loss
                last_improvement_eval = evals  # reset plateau tracker
                print(f"[SA] New global best at eval {evals}/{max_evals}: loss={best_loss:.6f}, temp={initial_temp / (i + 1):.6f}")
            else:
                # check for plateau
                if evals - last_improvement_eval > plateau_threshold:
                    print(f"[SA] Stopping early at eval {evals} due to plateau (improvement < {min_delta}) for {plateau_threshold}+ evals")
                    break

            # accept with probability via metropolis criterion
            diff = cand_loss - current_loss
            temp = initial_temp / (i + 1)  # linear decay
            if diff < 0 or torch.rand(1, device=device).item() < torch.exp(torch.tensor(-diff / temp, dtype=torch.float)):
                current_flat = candidate_flat.clone()
                current_loss = cand_loss

            if evals % 100 == 0:  # progress log every 100 evals
                print(f"[SA] Progress: eval {evals}/{max_evals}, current_loss={current_loss:.6f}, temp={temp:.6f}")

            i += 1

        # set final best params to model
        set_trainable_params(model, best_flat)
        print(f"[SA] Completed: total_evals={evals}, final_best_loss={best_loss:.6f}")

        step_info.update({
            'parameters': sum(p.numel() for p in model.parameters()),
            'max_evals': max_evals,
            'device': str(device),
            'initial_temp': initial_temp,
            'step_size': step_size,
            'decay': 'linear temp / (i+1)',
            'plateau_threshold': plateau_threshold,
            'min_delta': min_delta,
            'perturbation_distribution': 'gaussian N(0, step_size)',
            'evals': evals,
            'best_loss': best_loss,
            'wall_clock_time': time.perf_counter() - start_time,
            'history': history
        })
        return model, history
    

"""
Genetic Algorithm (GA) for Random Optimization.
Design: pop_size=50, tournament selection (fittest half), single-point crossover, gaussian mutation (rate=0.1, std=0.001), elitism.
Budget: stops at max_evals func evals.
Returns: optimized model, history [(evals, loss)] for analysis.
Optional: uniform plateau rule for early stopping if improvement < min_delta.
"""
def ga(
        model: nn.Module, 
        val_loader: torch.utils.data.DataLoader, 
        loss_fn: Callable, 
        device: torch.device, 
        max_evals: int = 10000, 
        pop_size: int = 50, 
        mutation_rate: float = 0.1, 
        mutation_std: float = 0.001, 
        plateau_threshold: int = 1000, 
        min_delta: float = 1e-6, 
        logger: MLLogger | None = None
    ) -> Tuple[nn.Module, List[Tuple[int, float]]]:
    print(f"[GA] Starting with {sum(p.numel() for p in model.parameters())} parameters, pop_size={pop_size}, max_evals={max_evals}")
    if logger is None: logger = MLLogger()
    start_time = time.perf_counter()
    last_improvement_eval = pop_size  # plateau tracking
    
    # create new individual with random perturbation
    def create_individual() -> torch.Tensor:
        return get_trainable_params(model) + torch.randn_like(get_trainable_params(model), device=device) * 0.1

    # single-point crossover
    def crossover(p1: torch.Tensor, p2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pivot = torch.randint(1, len(p1) - 1, (1,), device=device).item()
        return torch.cat((p1[:pivot], p2[pivot:])), torch.cat((p2[:pivot], p1[pivot:]))

    # apply gaussian mutation
    def mutate(ind: torch.Tensor) -> torch.Tensor:
        mask = torch.rand(len(ind), device=device) < mutation_rate
        return ind + mask.float() * torch.randn_like(ind, device=device) * mutation_std

    print(f"[GA] Initializing population of size {pop_size}")
    pop = [create_individual() for _ in range(pop_size)]
    fitness = [validation_objective(ind, model, val_loader, loss_fn, device) for ind in pop]  # minimize loss directly
    evals = pop_size
    prev_min_fit = float('inf')  # init prev to ensure first update
    history = [(evals, min(fitness))]  # track loss
    print(f"[GA] Initial population evaluated: evals={evals}, best_fitness={min(fitness):.6f}")

    with logger.log_step("Genetic Algorithms") as step_info:

        generation = 1
        while evals < max_evals:
            # sort by fitness (ascending for min loss)
            idx = torch.argsort(torch.tensor(fitness, device=device))
            pop = [pop[i] for i in idx]  # reorder pop
            fitness = [fitness[i] for i in idx]  # reorder fitness

            # elitism: keep best
            next_pop = [pop[0].clone()]

            # tournament selection and crossover
            for _ in range((pop_size - 1) // 2):
                idx1 = int(torch.randint(0, pop_size // 2, (1,), device=device).item())
                idx2 = int(torch.randint(0, pop_size // 2, (1,), device=device).item())
                p1, p2 = pop[idx1], pop[idx2]
                o1, o2 = crossover(p1, p2)
                o1 = mutate(o1)
                o2 = mutate(o2)
                next_pop.extend([o1, o2])

            # evaluate new population (skip elite)
            print(f"[GA] Evaluating generation {generation}: {len(next_pop) - 1} new individuals")
            new_fitness = [validation_objective(ind, model, val_loader, loss_fn, device) for ind in next_pop[1:]]
            evals += len(next_pop) - 1
            fitness = [fitness[0]] + new_fitness
            pop = next_pop

            min_fit = min(fitness)
            history.append((evals, min_fit))
            print(f"[GA] Generation {generation} complete: evals={evals}/{max_evals}, best_fitness={min_fit:.6f}")

            # check improvement delta
            improvement = prev_min_fit - min_fit  # positive if better
            if improvement > min_delta:
                last_improvement_eval = evals  # reset plateau tracker
            else:
                # check for plateau
                if evals - last_improvement_eval > plateau_threshold:
                    print(f"[GA] Stopping early at eval {evals} due to plateau (improvement < {min_delta}) for {plateau_threshold}+ evals")
                    break

            prev_min_fit = min_fit  # update prev for next check

            if evals >= max_evals:
                break
            generation += 1

        # set best individual to model
        best_idx = int(torch.argmin(torch.tensor(fitness, device=device)).item())
        set_trainable_params(model, pop[best_idx])
        print(f"[GA] Completed: total_evals={evals}, final_best_loss={min(fitness):.6f}")
        
        step_info.update({
            'parameters': sum(p.numel() for p in model.parameters()),
            'max_evals': max_evals,
            'device': str(device),
            'pop_size': pop_size,
            'mutation_rate': mutation_rate,
            'mutation_std': mutation_std,
            'selection': 'tournament (fittest half)',
            'crossover': 'single-point',
            'elitism': True,
            'mutation_distribution': 'gaussian std=0.001',
            'plateau_threshold': plateau_threshold,
            'min_delta': min_delta,
            'evals': evals,
            'best_loss': min(fitness),
            'wall_clock_time': time.perf_counter() - start_time,
            'history': history
        })
        return model, history


# example usage
if __name__ == "__main__":
    from models import MLP, set_seed
    set_seed(4242)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from data_processing import load_or_process_data, wrap_into_loaders
    X_train, X_val, X_test, y_train, y_val, y_test, info = load_or_process_data("hotels", "is_canceled", "classification", 0.1, 4242) # type: ignore
    train_loader, val_loader, test_loader = wrap_into_loaders("classification", X_train, X_val, X_test, y_train, y_val, y_test)

    # Initialize model
    model = MLP(in_dim=14, hidden=[128, 64], out_dim=4).to(device)
    model.freeze_all_but_last_k(k=2)  # freeze for ro part 1
    loss_fn = nn.CrossEntropyLoss()  # adjust for regression

    logger = MLLogger()

    # Run algorithms
    for algo in [rhc, sa, ga]:
        # Create a new model instance for each algo to avoid state overlap
        model_copy = MLP(in_dim=14, hidden=[128, 64], out_dim=4).to(device)
        model_copy.load_state_dict(model.state_dict())  # Copy weights
        model_copy.freeze_all_but_last_k(k=2)
        optimized_model, history = algo(model_copy, val_loader, loss_fn, device, max_evals=1000, logger=logger)
        print(f"{algo.__name__} history (evals, loss): {history[:5]}...")

        # Evaluate on validation data (using same loader as example)
        val_loss = validation_objective(get_trainable_params(optimized_model), optimized_model, val_loader, loss_fn, device)
        print(f"{algo.__name__} Validation Loss: {val_loss}")