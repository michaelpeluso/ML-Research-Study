# src/ro.py
# This file implements Randomized Hill Climbing (RHC), Simulated Annealing (SA), and Genetic Algorithm (GA)
# for continuous optimization of the frozen neural network parameters in Part 1.
# All implementations are adapted for minimization (validation loss) and track function evaluations to adhere to budgets.
# Design disclosures as per OL_Report_Fall_2025_v4.pdf:
# - RHC: Multiple restarts (default 10), step-size decay (linear from initial to 0.01*initial over iterations), Gaussian perturbation (std=step_size).
# - SA: Initial temperature=10.0, linear decay (temp / (i+1)), fixed step_size=0.1, Gaussian perturbation.
# - GA: Population size=50, fittest half selection, single-point crossover, Gaussian mutation (std=0.001, rate=0.1), elitism enabled.
# Budget: All stop at max_evals function evaluations (objective calls).
# Objective: Validation loss (include reg if applicable via model forward).
# Usage: Call with numpy array of flattened trainable params, objective function that takes np.array and returns float.

import numpy as np
from numpy.random import randn, rand, randint

def rhc(objective, init_state, max_evals, restarts=10, init_step_size=0.1):
    """
    Randomized Hill Climbing with restarts and step-size decay.
    """
    best = init_state.copy()
    best_eval = objective(best)
    evals = 1
    history = [(evals, best_eval)]

    for r in range(restarts):
        current = init_state + randn(len(init_state)) * init_step_size * 0.1  # Small perturbation for restart
        curr_eval = objective(current)
        evals += 1
        local_best = current.copy()
        local_best_eval = curr_eval

        n_iterations = (max_evals - evals) // restarts  # Approximate per restart
        for i in range(n_iterations):
            if evals >= max_evals:
                break
            step_size = init_step_size * (1 - i / n_iterations)  # Linear decay
            candidate = current + randn(len(init_state)) * step_size
            cand_eval = objective(candidate)
            evals += 1
            history.append((evals, local_best_eval))

            if cand_eval < local_best_eval:
                local_best = candidate.copy()
                local_best_eval = cand_eval

            if cand_eval < curr_eval:
                current = candidate.copy()
                curr_eval = cand_eval

        if local_best_eval < best_eval:
            best = local_best.copy()
            best_eval = local_best_eval

    return best, best_eval, history

def sa(objective, init_state, max_evals, step_size=0.1, temp=10.0):
    """
    Simulated Annealing adapted from machinelearningmastery.com.
    Modified to use max_evals instead of fixed iterations, no bounds.
    """
    best = init_state.copy()
    best_eval = objective(best)
    current = best.copy()
    curr_eval = best_eval
    evals = 1
    history = [(evals, best_eval)]

    i = 0
    while evals < max_evals:
        candidate = current + randn(len(init_state)) * step_size
        cand_eval = objective(candidate)
        evals += 1
        history.append((evals, best_eval))
        if evals >= max_evals:
            break

        if cand_eval < best_eval:
            best = candidate.copy()
            best_eval = cand_eval

        diff = cand_eval - curr_eval
        t = temp / (i + 1)  # Linear temperature decay
        metropolis = np.exp(-diff / t)
        if diff < 0 or rand() < metropolis:
            current = candidate.copy()
            curr_eval = cand_eval

        i += 1

    return best, best_eval, history

def ga(objective, init_state, max_evals, pop_size=50, mutation_rate=0.1, mutation_std=0.001):
    """
    Genetic Algorithm for continuous optimization, adapted and simplified from medium.com article.
    Maximizes but inverted for minimization (negative objective).
    Uses fittest half selection, single-point crossover, Gaussian mutation, elitism.
    """
    def create_individual(dim, scale=0.1):
        return init_state + np.random.randn(dim) * scale

    def crossover(p1, p2):
        pivot = randint(1, len(p1) - 1)
        return np.concatenate((p1[:pivot], p2[pivot:])), np.concatenate((p2[:pivot], p1[pivot:]))

    def mutate(ind, rate, std):
        for i in range(len(ind)):
            if rand() < rate:
                ind[i] += np.random.randn() * std
        return ind

    dim = len(init_state)
    pop = [create_individual(dim) for _ in range(pop_size - 1)]
    pop.append(init_state.copy())  # Include init as elite
    fitness = [-objective(ind) for ind in pop]  # Negative for maximization
    evals = pop_size
    history = [(evals, -max(fitness))]  # Min loss = -max(-loss)

    while evals < max_evals:
        # Sort by fitness (descending for max)
        sorted_idx = np.argsort(fitness)[::-1]
        pop = [pop[i] for i in sorted_idx]
        fitness = [fitness[i] for i in sorted_idx]

        # Elitism: keep best
        next_pop = [pop[0].copy()]

        # Select fittest half
        selected = pop[:pop_size // 2]

        # Create offspring
        for _ in range((pop_size - 1) // 2):
            p1, p2 = selected[randint(0, len(selected) - 1)], selected[randint(0, len(selected) - 1)]
            o1, o2 = crossover(p1, p2)
            o1 = mutate(o1, mutation_rate, mutation_std)
            o2 = mutate(o2, mutation_rate, mutation_std)
            next_pop.extend([o1, o2])

        # Evaluate new pop (skip elite if unchanged)
        new_fitness = [-objective(ind) for ind in next_pop[1:]]
        evals += len(next_pop) - 1
        fitness = [-objective(next_pop[0])] + new_fitness
        pop = next_pop

        history.append((evals, -max(fitness)))  # Track best min loss

        if evals >= max_evals:
            break

    best_idx = np.argmax(fitness)
    return pop[best_idx], -fitness[best_idx], history