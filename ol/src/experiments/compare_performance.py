"""
Performance Comparison Script
==============================
Compares three key stages to show accuracy progression:
1. PyTorch Default Parameters (baseline)
2. Tuned Backbone (from main.py - no optimization)
3. Individual Optimizations (Parts 1, 2, 3 - best from each)

This shows the incremental improvement at each stage.

Usage:
    python compare_performance.py --dataset hotels --seeds 42,4242,424242
    python compare_performance.py --dataset accidents --quick
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn

from core.models import MLP, set_seed
from core.random_optimizers import rhc, sa, ga
from core.training import eval_loss
from utils.data_processing import load_or_process_data, wrap_into_loaders


# =============================================================================
# HELPER FUNCTION
# =============================================================================

def train_simple(model, train_loader, val_loader, loss_fn, optimizer, max_iter, device):
    """Simple training loop returning history"""
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(max_iter):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            if output.shape[-1] == 1:
                output = output.squeeze(-1)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = epoch_loss / n_batches
        val_loss = eval_loss(model, val_loader, loss_fn, device)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
    
    return history


# =============================================================================
# CONFIGURATION
# =============================================================================

TUNED_PARAMS = {
    'hotels': {
        'nn_2': {
            'max_iter': 15,
            'learning_rate': 1e-03,  # Fixed: main.py uses alpha (1e-05) as learning_rate
            'hidden_layers': [256, 128],
            'weight_decay': 1e-05,
            'activation': 'tanh',
            'batch_size': 64,
        },
        'nn_4': {
            'max_iter': 15,
            'learning_rate': 1e-05,  # Fixed: main.py uses alpha (1e-05) as learning_rate
            'hidden_layers': [256, 256, 128, 128],
            'weight_decay': 1e-03,
            'activation': 'tanh',
            'batch_size': 64,
        }
    },
    'accidents': {
        'nn_2': {
            'max_iter': 5,
            'learning_rate': 0.001,  # Fixed: main.py uses alpha (0.001) as learning_rate
            'hidden_layers': [256, 128],
            'weight_decay': 0.001,
            'activation': 'relu',
            'batch_size': 1024,
        },
        'nn_4': {
            'max_iter': 5,
            'learning_rate': 0.001,  # Fixed: main.py uses alpha (0.001) as learning_rate
            'hidden_layers': [256, 256, 128, 128],
            'weight_decay': 0.001,
            'activation': 'relu',
            'batch_size': 1024,
        }
    }
}


# =============================================================================
# STAGE 1: PYTORCH DEFAULTS
# =============================================================================

def run_stage1_pytorch_defaults(dataset, method, train_loader, val_loader, test_loader,
                                 in_dim, out_dim, device, seeds):
    """Stage 1: PyTorch default parameters (lr=0.001, no tuning)"""
    print("\n" + "="*80)
    print(f"STAGE 1: PYTORCH DEFAULT PARAMETERS - {dataset.upper()}")
    print("="*80)
    
    results = {}
    loss_fn = nn.CrossEntropyLoss() if method == 'classification' else nn.MSELoss()
    
    for arch_name, hidden in [('nn_2', [256, 128]), ('nn_4', [256, 256, 128, 128])]:
        print(f"\n{arch_name}:")
        seed_results = []
        
        for seed in seeds:
            set_seed(seed)
            
            # Default PyTorch: ReLU, lr=0.001, no weight decay
            model = MLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim, activation='relu').to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Train for 15 epochs (standard)
            history = train_simple(model, train_loader, val_loader, loss_fn, optimizer, 15, device)
            
            val_loss = history['val_loss'][-1]
            test_loss = eval_loss(model, test_loader, loss_fn, device)
            
            seed_results.append({'val': val_loss, 'test': test_loss})
            print(f"  Seed {seed}: Val={val_loss:.6f}, Test={test_loss:.6f}")
        
        results[arch_name] = {
            'mean_val': np.mean([r['val'] for r in seed_results]),
            'std_val': np.std([r['val'] for r in seed_results]),
            'mean_test': np.mean([r['test'] for r in seed_results]),
            'std_test': np.std([r['test'] for r in seed_results]),
        }
        
        print(f"  Average: Val={results[arch_name]['mean_val']:.6f}±{results[arch_name]['std_val']:.6f}, "
              f"Test={results[arch_name]['mean_test']:.6f}±{results[arch_name]['std_test']:.6f}")
    
    return results


# =============================================================================
# STAGE 2: TUNED BACKBONE
# =============================================================================

def run_stage2_tuned_backbone(dataset, method, train_loader, val_loader, test_loader,
                               in_dim, out_dim, device, seeds):
    """Stage 2: Tuned hyperparameters from main.py (no optimization yet)"""
    print("\n" + "="*80)
    print(f"STAGE 2: TUNED BACKBONE (NO OPTIMIZATION) - {dataset.upper()}")
    print("="*80)
    
    results = {}
    loss_fn = nn.CrossEntropyLoss() if method == 'classification' else nn.MSELoss()
    
    for arch_name in ['nn_2', 'nn_4']:
        params = TUNED_PARAMS[dataset][arch_name]
        print(f"\n{arch_name} (LR={params['learning_rate']}, Activation={params['activation']}):")
        seed_results = []
        
        for seed in seeds:
            set_seed(seed)
            
            model = MLP(
                in_dim=in_dim,
                hidden=params['hidden_layers'],
                out_dim=out_dim,
                activation=params['activation']
            ).to(device)
            
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
            
            history = train_simple(model, train_loader, val_loader, loss_fn, optimizer,
                                  params['max_iter'], device)
            
            val_loss = history['val_loss'][-1]
            test_loss = eval_loss(model, test_loader, loss_fn, device)
            
            seed_results.append({'val': val_loss, 'test': test_loss})
            print(f"  Seed {seed}: Val={val_loss:.6f}, Test={test_loss:.6f}")
        
        results[arch_name] = {
            'mean_val': np.mean([r['val'] for r in seed_results]),
            'std_val': np.std([r['val'] for r in seed_results]),
            'mean_test': np.mean([r['test'] for r in seed_results]),
            'std_test': np.std([r['test'] for r in seed_results]),
        }
        
        print(f"  Average: Val={results[arch_name]['mean_val']:.6f}±{results[arch_name]['std_val']:.6f}, "
              f"Test={results[arch_name]['mean_test']:.6f}±{results[arch_name]['std_test']:.6f}")
    
    return results


# =============================================================================
# STAGE 3: INDIVIDUAL OPTIMIZATIONS
# =============================================================================

def run_stage3_part1_ro(dataset, method, train_loader, val_loader, test_loader,
                        in_dim, out_dim, device, seeds, max_evals):
    """Stage 3a: Random Optimization (best of RHC/SA/GA)"""
    print("\n" + "="*80)
    print(f"STAGE 3 - PART 1: RANDOM OPTIMIZATION - {dataset.upper()}")
    print("="*80)
    
    results = {}
    loss_fn = nn.CrossEntropyLoss() if method == 'classification' else nn.MSELoss()
    
    for arch_name in ['nn_2', 'nn_4']:
        params = TUNED_PARAMS[dataset][arch_name]
        print(f"\n{arch_name}:")
        
        # Test all three algorithms
        algo_results = {}
        for algo_name, algo_fn, algo_params in [
            ('RHC', rhc, {'restarts': 5, 'max_evals': max_evals, 'initial_perturb_scale': 0.1,
                          'decay_rate': 0.995, 'plateau_threshold': 250}),
            ('SA', sa, {'max_evals': max_evals, 'initial_temp': 0.1, 'min_temp': 0.001,
                        'cooling_rate': 0.003, 'initial_perturb_scale': 0.1,
                        'perturb_decay': 0.995, 'plateau_threshold': 500}),
            ('GA', ga, {'max_evals': max_evals, 'pop_size': 20, 'mutation_rate': 0.1,
                        'mutation_std': 0.01, 'plateau_threshold': 250})
        ]:
            seed_results = []
            
            for seed in seeds:
                set_seed(seed)
                
                # Pre-train with tuned params
                model = MLP(in_dim=in_dim, hidden=params['hidden_layers'],
                           out_dim=out_dim, activation=params['activation']).to(device)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                            weight_decay=params['weight_decay'])
                train_simple(model, train_loader, val_loader, loss_fn, optimizer,
                           params['max_iter'], device)
                
                # Freeze and optimize
                model.freeze_all_but_last_k(k=1, limit=50000)
                optimized_model, history = algo_fn(model, val_loader, loss_fn, device, **algo_params)
                
                test_loss = eval_loss(optimized_model, test_loader, loss_fn, device)
                seed_results.append(test_loss)
            
            algo_results[algo_name] = {
                'mean_test': np.mean(seed_results),
                'std_test': np.std(seed_results)
            }
            print(f"  {algo_name}: Test={algo_results[algo_name]['mean_test']:.6f}±"
                  f"{algo_results[algo_name]['std_test']:.6f}")
        
        # Pick best algorithm
        best_algo = min(algo_results.items(), key=lambda x: x[1]['mean_test'])
        results[arch_name] = {
            'best_algo': best_algo[0],
            'mean_test': best_algo[1]['mean_test'],
            'std_test': best_algo[1]['std_test'],
            'all_algos': algo_results
        }
        print(f"  → Best: {best_algo[0]}")
    
    return results


def run_stage3_part2_adam_ablations(dataset, method, train_loader, val_loader, test_loader,
                                     in_dim, out_dim, device, seeds):
    """Stage 3b: Adam Ablations (best optimizer variant)"""
    print("\n" + "="*80)
    print(f"STAGE 3 - PART 2: ADAM ABLATIONS - {dataset.upper()}")
    print("="*80)
    
    results = {}
    loss_fn = nn.CrossEntropyLoss() if method == 'classification' else nn.MSELoss()
    
    for arch_name in ['nn_2', 'nn_4']:
        params = TUNED_PARAMS[dataset][arch_name]
        print(f"\n{arch_name}:")
        
        opt_results = {}
        for opt_name, opt_fn in [
            ('Adam', lambda p, lr: torch.optim.Adam(p, lr=lr)),
            ('AdamW', lambda p, lr: torch.optim.AdamW(p, lr=lr, weight_decay=0.01)),
            ('SGD+Momentum', lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9)),
        ]:
            seed_results = []
            
            for seed in seeds:
                set_seed(seed)
                
                model = MLP(in_dim=in_dim, hidden=params['hidden_layers'],
                           out_dim=out_dim, activation=params['activation']).to(device)
                optimizer = opt_fn(model.parameters(), params['learning_rate'])
                
                train_simple(model, train_loader, val_loader, loss_fn, optimizer,
                           params['max_iter'], device)
                
                test_loss = eval_loss(model, test_loader, loss_fn, device)
                seed_results.append(test_loss)
            
            opt_results[opt_name] = {
                'mean_test': np.mean(seed_results),
                'std_test': np.std(seed_results)
            }
            print(f"  {opt_name}: Test={opt_results[opt_name]['mean_test']:.6f}±"
                  f"{opt_results[opt_name]['std_test']:.6f}")
        
        best_opt = min(opt_results.items(), key=lambda x: x[1]['mean_test'])
        results[arch_name] = {
            'best_optimizer': best_opt[0],
            'mean_test': best_opt[1]['mean_test'],
            'std_test': best_opt[1]['std_test'],
            'all_optimizers': opt_results
        }
        print(f"  → Best: {best_opt[0]}")
    
    return results


def run_stage3_part3_regularization(dataset, method, train_loader, val_loader, test_loader,
                                     in_dim, out_dim, device, seeds):
    """Stage 3c: Regularization (best config)"""
    print("\n" + "="*80)
    print(f"STAGE 3 - PART 3: REGULARIZATION - {dataset.upper()}")
    print("="*80)
    
    results = {}
    loss_fn = nn.CrossEntropyLoss() if method == 'classification' else nn.MSELoss()
    
    for arch_name in ['nn_2', 'nn_4']:
        params = TUNED_PARAMS[dataset][arch_name]
        print(f"\n{arch_name}:")
        
        reg_results = {}
        for config_name, config in [
            ('No Reg', {}),
            ('L2 (0.001)', {'weight_decay': 0.001}),
            ('Dropout (0.2)', {'dropout': 0.2}),
        ]:
            seed_results = []
            
            for seed in seeds:
                set_seed(seed)
                
                if 'dropout' in config:
                    model = MLP(in_dim=in_dim, hidden=params['hidden_layers'],
                               out_dim=out_dim, activation=params['activation'],
                               dropout_p=config['dropout']).to(device)
                else:
                    model = MLP(in_dim=in_dim, hidden=params['hidden_layers'],
                               out_dim=out_dim, activation=params['activation']).to(device)
                
                wd = config.get('weight_decay', params['weight_decay'])
                optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                            weight_decay=wd)
                
                train_simple(model, train_loader, val_loader, loss_fn, optimizer,
                           params['max_iter'], device)
                
                test_loss = eval_loss(model, test_loader, loss_fn, device)
                seed_results.append(test_loss)
            
            reg_results[config_name] = {
                'mean_test': np.mean(seed_results),
                'std_test': np.std(seed_results)
            }
            print(f"  {config_name}: Test={reg_results[config_name]['mean_test']:.6f}±"
                  f"{reg_results[config_name]['std_test']:.6f}")
        
        best_reg = min(reg_results.items(), key=lambda x: x[1]['mean_test'])
        results[arch_name] = {
            'best_config': best_reg[0],
            'mean_test': best_reg[1]['mean_test'],
            'std_test': best_reg[1]['std_test'],
            'all_configs': reg_results
        }
        print(f"  → Best: {best_reg[0]}")
    
    return results


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def main_comparison():
    parser = argparse.ArgumentParser(description='Compare performance across stages')
    parser.add_argument('--dataset', type=str, required=True, choices=['hotels', 'accidents'])
    parser.add_argument('--seeds', type=str, default='42,4242,424242')
    parser.add_argument('--quick', action='store_true', help='Reduced budgets for testing')
    
    args = parser.parse_args()
    
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    max_evals = 1000 if args.quick else 10000
    
    # Configuration
    if args.dataset == 'hotels':
        target = 'is_canceled'
        method = 'classification'
    else:
        target = 'Duration_Seconds'
        method = 'regression'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, info = load_or_process_data(
        dataset=args.dataset,
        method=method,
        target=target,
        subsample=0.1,
        seed=seeds[0]
    )
    
    batch_size = TUNED_PARAMS[args.dataset]['nn_2']['batch_size']
    train_loader, val_loader, test_loader = wrap_into_loaders(
        method=method,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        batch_size=batch_size
    )
    
    in_dim = X_train.shape[1]
    out_dim = len(y_train.unique()) if method == 'classification' else 1
    
    print(f"Dataset: {args.dataset}")
    print(f"Input dim: {in_dim}, Output dim: {out_dim}")
    print(f"Seeds: {seeds}")
    
    # Run all stages
    all_results = {}
    
    all_results['stage1_defaults'] = run_stage1_pytorch_defaults(
        args.dataset, method, train_loader, val_loader, test_loader,
        in_dim, out_dim, device, seeds
    )
    
    all_results['stage2_tuned'] = run_stage2_tuned_backbone(
        args.dataset, method, train_loader, val_loader, test_loader,
        in_dim, out_dim, device, seeds
    )
    
    all_results['stage3_part1_ro'] = run_stage3_part1_ro(
        args.dataset, method, train_loader, val_loader, test_loader,
        in_dim, out_dim, device, seeds, max_evals
    )
    
    all_results['stage3_part2_adam'] = run_stage3_part2_adam_ablations(
        args.dataset, method, train_loader, val_loader, test_loader,
        in_dim, out_dim, device, seeds
    )
    
    all_results['stage3_part3_reg'] = run_stage3_part3_regularization(
        args.dataset, method, train_loader, val_loader, test_loader,
        in_dim, out_dim, device, seeds
    )
    
    # Save results
    output_dir = Path(os.environ['ROOT']) / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f'{args.dataset}_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary table
    print("\n" + "="*100)
    print("ACCURACY PROGRESSION SUMMARY")
    print("="*100)
    print(f"{'Stage':<40} {'nn_2 Test Loss':<25} {'nn_4 Test Loss':<25}")
    print("-"*100)
    
    for arch in ['nn_2', 'nn_4']:
        stage1 = all_results['stage1_defaults'][arch]
        stage2 = all_results['stage2_tuned'][arch]
        stage3_ro = all_results['stage3_part1_ro'][arch]
        stage3_adam = all_results['stage3_part2_adam'][arch]
        stage3_reg = all_results['stage3_part3_reg'][arch]
        
        if arch == 'nn_2':
            print(f"{'Stage 1: PyTorch Defaults':<40} "
                  f"{stage1['mean_test']:.6f}±{stage1['std_test']:.6f}      "
                  f"{all_results['stage1_defaults']['nn_4']['mean_test']:.6f}±"
                  f"{all_results['stage1_defaults']['nn_4']['std_test']:.6f}")
            
            print(f"{'Stage 2: Tuned Backbone':<40} "
                  f"{stage2['mean_test']:.6f}±{stage2['std_test']:.6f}      "
                  f"{all_results['stage2_tuned']['nn_4']['mean_test']:.6f}±"
                  f"{all_results['stage2_tuned']['nn_4']['std_test']:.6f}")
            
            print(f"{'Stage 3a: Best RO (' + stage3_ro['best_algo'] + ')':<40} "
                  f"{stage3_ro['mean_test']:.6f}±{stage3_ro['std_test']:.6f}      "
                  f"{all_results['stage3_part1_ro']['nn_4']['mean_test']:.6f}±"
                  f"{all_results['stage3_part1_ro']['nn_4']['std_test']:.6f}")
            
            print(f"{'Stage 3b: Best Opt (' + stage3_adam['best_optimizer'] + ')':<40} "
                  f"{stage3_adam['mean_test']:.6f}±{stage3_adam['std_test']:.6f}      "
                  f"{all_results['stage3_part2_adam']['nn_4']['mean_test']:.6f}±"
                  f"{all_results['stage3_part2_adam']['nn_4']['std_test']:.6f}")
            
            print(f"{'Stage 3c: Best Reg (' + stage3_reg['best_config'] + ')':<40} "
                  f"{stage3_reg['mean_test']:.6f}±{stage3_reg['std_test']:.6f}      "
                  f"{all_results['stage3_part3_reg']['nn_4']['mean_test']:.6f}±"
                  f"{all_results['stage3_part3_reg']['nn_4']['std_test']:.6f}")
    
    print("="*100)
    print(f"\nResults saved to: {output_dir / f'{args.dataset}_comparison.json'}")
    print("="*100)
    
    # Generate execution report in figures folder
    report_dir = Path(os.environ['ROOT']) / 'figures' / args.dataset / 'comparison'
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / 'comparison_execution_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("PERFORMANCE COMPARISON EXECUTION REPORT\n")
        f.write(f"Dataset: {args.dataset.upper()}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Quick Mode: {args.quick}\n")
        f.write(f"Max Evals (Part 1): {max_evals}\n")
        f.write("="*100 + "\n\n")
        
        f.write("ACCURACY PROGRESSION SUMMARY\n")
        f.write("="*100 + "\n")
        f.write(f"{'Stage':<40} {'nn_2 Test Loss':<25} {'nn_4 Test Loss':<25}\n")
        f.write("-"*100 + "\n")
        
        stage1_nn2 = all_results['stage1_defaults']['nn_2']
        stage1_nn4 = all_results['stage1_defaults']['nn_4']
        stage2_nn2 = all_results['stage2_tuned']['nn_2']
        stage2_nn4 = all_results['stage2_tuned']['nn_4']
        stage3_ro_nn2 = all_results['stage3_part1_ro']['nn_2']
        stage3_ro_nn4 = all_results['stage3_part1_ro']['nn_4']
        stage3_adam_nn2 = all_results['stage3_part2_adam']['nn_2']
        stage3_adam_nn4 = all_results['stage3_part2_adam']['nn_4']
        stage3_reg_nn2 = all_results['stage3_part3_reg']['nn_2']
        stage3_reg_nn4 = all_results['stage3_part3_reg']['nn_4']
        
        f.write(f"{'Stage 1: PyTorch Defaults':<40} "
                f"{stage1_nn2['mean_test']:.6f}±{stage1_nn2['std_test']:.6f}      "
                f"{stage1_nn4['mean_test']:.6f}±{stage1_nn4['std_test']:.6f}\n")
        
        f.write(f"{'Stage 2: Tuned Backbone':<40} "
                f"{stage2_nn2['mean_test']:.6f}±{stage2_nn2['std_test']:.6f}      "
                f"{stage2_nn4['mean_test']:.6f}±{stage2_nn4['std_test']:.6f}\n")
        
        f.write(f"{'Stage 3a: Best RO (' + stage3_ro_nn2['best_algo'] + ')':<40} "
                f"{stage3_ro_nn2['mean_test']:.6f}±{stage3_ro_nn2['std_test']:.6f}      "
                f"{stage3_ro_nn4['mean_test']:.6f}±{stage3_ro_nn4['std_test']:.6f}\n")
        
        f.write(f"{'Stage 3b: Best Opt (' + stage3_adam_nn2['best_optimizer'] + ')':<40} "
                f"{stage3_adam_nn2['mean_test']:.6f}±{stage3_adam_nn2['std_test']:.6f}      "
                f"{stage3_adam_nn4['mean_test']:.6f}±{stage3_adam_nn4['std_test']:.6f}\n")
        
        f.write(f"{'Stage 3c: Best Reg (' + stage3_reg_nn2['best_config'] + ')':<40} "
                f"{stage3_reg_nn2['mean_test']:.6f}±{stage3_reg_nn2['std_test']:.6f}      "
                f"{stage3_reg_nn4['mean_test']:.6f}±{stage3_reg_nn4['std_test']:.6f}\n")
        
        f.write("="*100 + "\n\n")
        
        # Detailed results by stage
        f.write("DETAILED RESULTS BY STAGE\n")
        f.write("="*100 + "\n\n")
        
        for stage_name, stage_data in [
            ("Stage 1: PyTorch Defaults", all_results['stage1_defaults']),
            ("Stage 2: Tuned Backbone", all_results['stage2_tuned']),
        ]:
            f.write(f"{stage_name}\n")
            f.write("-"*100 + "\n")
            for arch in ['nn_2', 'nn_4']:
                f.write(f"  {arch}:\n")
                f.write(f"    Val Loss: {stage_data[arch]['mean_val']:.6f} ± {stage_data[arch]['std_val']:.6f}\n")
                f.write(f"    Test Loss: {stage_data[arch]['mean_test']:.6f} ± {stage_data[arch]['std_test']:.6f}\n")
            f.write("\n")
        
        # Part 1: Random Optimization
        f.write("Stage 3a: Random Optimization\n")
        f.write("-"*100 + "\n")
        for arch in ['nn_2', 'nn_4']:
            f.write(f"  {arch}:\n")
            f.write(f"    Best Algorithm: {all_results['stage3_part1_ro'][arch]['best_algo']}\n")
            f.write(f"    Test Loss: {all_results['stage3_part1_ro'][arch]['mean_test']:.6f} ± "
                   f"{all_results['stage3_part1_ro'][arch]['std_test']:.6f}\n")
            f.write(f"    All Algorithms:\n")
            for algo, result in all_results['stage3_part1_ro'][arch]['all_algos'].items():
                f.write(f"      {algo}: {result['mean_test']:.6f} ± {result['std_test']:.6f}\n")
        f.write("\n")
        
        # Part 2: Adam Ablations
        f.write("Stage 3b: Adam Ablations\n")
        f.write("-"*100 + "\n")
        for arch in ['nn_2', 'nn_4']:
            f.write(f"  {arch}:\n")
            f.write(f"    Best Optimizer: {all_results['stage3_part2_adam'][arch]['best_optimizer']}\n")
            f.write(f"    Test Loss: {all_results['stage3_part2_adam'][arch]['mean_test']:.6f} ± "
                   f"{all_results['stage3_part2_adam'][arch]['std_test']:.6f}\n")
            f.write(f"    All Optimizers:\n")
            for opt, result in all_results['stage3_part2_adam'][arch]['all_optimizers'].items():
                f.write(f"      {opt}: {result['mean_test']:.6f} ± {result['std_test']:.6f}\n")
        f.write("\n")
        
        # Part 3: Regularization
        f.write("Stage 3c: Targeted Regularization\n")
        f.write("-"*100 + "\n")
        for arch in ['nn_2', 'nn_4']:
            f.write(f"  {arch}:\n")
            f.write(f"    Best Config: {all_results['stage3_part3_reg'][arch]['best_config']}\n")
            f.write(f"    Test Loss: {all_results['stage3_part3_reg'][arch]['mean_test']:.6f} ± "
                   f"{all_results['stage3_part3_reg'][arch]['std_test']:.6f}\n")
            f.write(f"    All Configs:\n")
            for cfg, result in all_results['stage3_part3_reg'][arch]['all_configs'].items():
                f.write(f"      {cfg}: {result['mean_test']:.6f} ± {result['std_test']:.6f}\n")
        f.write("\n")
        
        f.write("="*100 + "\n")
        f.write(f"JSON results: {output_dir / f'{args.dataset}_comparison.json'}\n")
        f.write("="*100 + "\n")
    
    print(f"\nExecution report saved to: {report_path}")


# =============================================================================
# CALLABLE FUNCTION FOR main.py
# =============================================================================

def generate_comparison_report(exp, architecture: str, seeds: List[int] = [42, 4242, 424242],
                               quick: bool = False):
    """
    Generate comparison report for a single experiment/architecture combination.
    Called from main.py to create reports in figures/{dataset}/{architecture}/
    """
    dataset = exp.dataset
    save_path = exp.save_path
    os.makedirs(save_path, exist_ok=True)
    max_evals = 1000 if quick else 10000
    
    print(f"\n{'='*100}")
    print(f"GENERATING COMPARISON REPORT: {dataset.upper()} - {architecture}")
    print(f"{'='*100}")
    
    # Get data loaders
    train_loader, val_loader, test_loader = exp.get_data()
    in_dim = train_loader.dataset.tensors[0].shape[1]
    out_dim = len(train_loader.dataset.tensors[1].unique()) if exp.method == 'classification' else 1
    
    # Get tuned parameters for this dataset/architecture
    params = TUNED_PARAMS[dataset][architecture]
    loss_fn = nn.CrossEntropyLoss() if exp.method == 'classification' else nn.MSELoss()
    device = exp.device
    method = exp.method
    
    all_results = {}
    
    # Stage 1: PyTorch Defaults
    print(f"\nStage 1: PyTorch Defaults")
    seed_results = []
    for seed in seeds:
        set_seed(seed)
        model = MLP(in_dim=in_dim, hidden=params['hidden_layers'], out_dim=out_dim, 
                   activation='relu').to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_simple(model, train_loader, val_loader, loss_fn, optimizer, 15, device)
        
        val_loss = eval_loss(model, val_loader, loss_fn, device)
        test_loss = eval_loss(model, test_loader, loss_fn, device)
        seed_results.append({'val': val_loss, 'test': test_loss})
        print(f"  Seed {seed}: Val={val_loss:.6f}, Test={test_loss:.6f}")
    
    all_results['stage1_defaults'] = {
        'mean_val': np.mean([r['val'] for r in seed_results]),
        'std_val': np.std([r['val'] for r in seed_results]),
        'mean_test': np.mean([r['test'] for r in seed_results]),
        'std_test': np.std([r['test'] for r in seed_results]),
    }
    
    # Stage 2: Tuned Backbone
    print(f"\nStage 2: Tuned Backbone")
    seed_results = []
    for seed in seeds:
        set_seed(seed)
        model = MLP(in_dim=in_dim, hidden=params['hidden_layers'], out_dim=out_dim,
                   activation=params['activation']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                     weight_decay=params['weight_decay'])
        train_simple(model, train_loader, val_loader, loss_fn, optimizer, 
                    params['max_iter'], device)
        
        val_loss = eval_loss(model, val_loader, loss_fn, device)
        test_loss = eval_loss(model, test_loader, loss_fn, device)
        seed_results.append({'val': val_loss, 'test': test_loss})
        print(f"  Seed {seed}: Val={val_loss:.6f}, Test={test_loss:.6f}")
    
    all_results['stage2_tuned'] = {
        'mean_val': np.mean([r['val'] for r in seed_results]),
        'std_val': np.std([r['val'] for r in seed_results]),
        'mean_test': np.mean([r['test'] for r in seed_results]),
        'std_test': np.std([r['test'] for r in seed_results]),
    }
    
    # Stage 3a: Random Optimization
    print(f"\nStage 3a: Random Optimization")
    algo_results = {}
    for algo_name, algo_fn, algo_params in [
        ('RHC', rhc, {'restarts': 5, 'max_evals': max_evals, 'initial_perturb_scale': 0.1,
                      'decay_rate': 0.995, 'plateau_threshold': 250}),
        ('SA', sa, {'max_evals': max_evals, 'initial_temp': 0.1, 'min_temp': 0.001,
                    'cooling_rate': 0.003, 'initial_perturb_scale': 0.1,
                    'perturb_decay': 0.995, 'plateau_threshold': 500}),
        ('GA', ga, {'max_evals': max_evals, 'pop_size': 20, 'mutation_rate': 0.1,
                    'mutation_std': 0.01, 'plateau_threshold': 250})
    ]:
        seed_results = []
        for seed in seeds:
            set_seed(seed)
            model = MLP(in_dim=in_dim, hidden=params['hidden_layers'], out_dim=out_dim,
                       activation=params['activation']).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                         weight_decay=params['weight_decay'])
            train_simple(model, train_loader, val_loader, loss_fn, optimizer,
                        params['max_iter'], device)
            
            model.freeze_all_but_last_k(k=1, limit=50000)
            optimized_model, history = algo_fn(model, val_loader, loss_fn, device, **algo_params)
            test_loss = eval_loss(optimized_model, test_loader, loss_fn, device)
            seed_results.append(test_loss)
        
        algo_results[algo_name] = {
            'mean_test': np.mean(seed_results),
            'std_test': np.std(seed_results)
        }
        print(f"  {algo_name}: Test={algo_results[algo_name]['mean_test']:.6f}±{algo_results[algo_name]['std_test']:.6f}")
    
    best_algo = min(algo_results.items(), key=lambda x: x[1]['mean_test'])
    all_results['stage3_part1_ro'] = {
        'best_algo': best_algo[0],
        'mean_test': best_algo[1]['mean_test'],
        'std_test': best_algo[1]['std_test'],
        'all_algos': algo_results
    }
    
    # Stage 3b: Adam Ablations
    print(f"\nStage 3b: Adam Ablations")
    opt_results = {}
    for opt_name, opt_fn in [
        ('Adam', lambda p, lr: torch.optim.Adam(p, lr=lr)),
        ('AdamW', lambda p, lr: torch.optim.AdamW(p, lr=lr, weight_decay=0.01)),
        ('SGD+Momentum', lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9)),
    ]:
        seed_results = []
        for seed in seeds:
            set_seed(seed)
            model = MLP(in_dim=in_dim, hidden=params['hidden_layers'], out_dim=out_dim,
                       activation=params['activation']).to(device)
            optimizer = opt_fn(model.parameters(), params['learning_rate'])
            train_simple(model, train_loader, val_loader, loss_fn, optimizer,
                        params['max_iter'], device)
            test_loss = eval_loss(model, test_loader, loss_fn, device)
            seed_results.append(test_loss)
        
        opt_results[opt_name] = {
            'mean_test': np.mean(seed_results),
            'std_test': np.std(seed_results)
        }
        print(f"  {opt_name}: Test={opt_results[opt_name]['mean_test']:.6f}±{opt_results[opt_name]['std_test']:.6f}")
    
    best_opt = min(opt_results.items(), key=lambda x: x[1]['mean_test'])
    all_results['stage3_part2_adam'] = {
        'best_optimizer': best_opt[0],
        'mean_test': best_opt[1]['mean_test'],
        'std_test': best_opt[1]['std_test'],
        'all_optimizers': opt_results
    }
    
    # Stage 3c: Regularization
    print(f"\nStage 3c: Targeted Regularization")
    reg_results = {}
    for config_name, config in [
        ('No Reg', {}),
        ('L2 (0.001)', {'weight_decay': 0.001}),
        ('Dropout (0.2)', {'dropout': 0.2}),
    ]:
        seed_results = []
        for seed in seeds:
            set_seed(seed)
            if 'dropout' in config:
                model = MLP(in_dim=in_dim, hidden=params['hidden_layers'], out_dim=out_dim,
                           activation=params['activation'], dropout_p=config['dropout']).to(device)
            else:
                model = MLP(in_dim=in_dim, hidden=params['hidden_layers'], out_dim=out_dim,
                           activation=params['activation']).to(device)
            
            wd = config.get('weight_decay', params['weight_decay'])
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                        weight_decay=wd)
            train_simple(model, train_loader, val_loader, loss_fn, optimizer,
                        params['max_iter'], device)
            test_loss = eval_loss(model, test_loader, loss_fn, device)
            seed_results.append(test_loss)
        
        reg_results[config_name] = {
            'mean_test': np.mean(seed_results),
            'std_test': np.std(seed_results)
        }
        print(f"  {config_name}: Test={reg_results[config_name]['mean_test']:.6f}±{reg_results[config_name]['std_test']:.6f}")
    
    best_reg = min(reg_results.items(), key=lambda x: x[1]['mean_test'])
    all_results['stage3_part3_reg'] = {
        'best_config': best_reg[0],
        'mean_test': best_reg[1]['mean_test'],
        'std_test': best_reg[1]['std_test'],
        'all_configs': reg_results
    }
    
    # Save JSON results
    json_path = Path(save_path) / 'comparison_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate text report
    report_path = Path(save_path) / 'comparison_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write(f"PERFORMANCE COMPARISON REPORT: {dataset.upper()} - {architecture}\n")
        f.write("="*100 + "\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Quick Mode: {quick}\n")
        f.write("\n")
        
        f.write("ACCURACY PROGRESSION SUMMARY\n")
        f.write("-"*100 + "\n\n")
        
        f.write("Stage 1: PyTorch Defaults\n")
        f.write(f"  Val Loss:  {all_results['stage1_defaults']['mean_val']:.6f} ± {all_results['stage1_defaults']['std_val']:.6f}\n")
        f.write(f"  Test Loss: {all_results['stage1_defaults']['mean_test']:.6f} ± {all_results['stage1_defaults']['std_test']:.6f}\n\n")
        
        f.write("Stage 2: Tuned Backbone\n")
        f.write(f"  Val Loss:  {all_results['stage2_tuned']['mean_val']:.6f} ± {all_results['stage2_tuned']['std_val']:.6f}\n")
        f.write(f"  Test Loss: {all_results['stage2_tuned']['mean_test']:.6f} ± {all_results['stage2_tuned']['std_test']:.6f}\n")
        improvement = ((all_results['stage1_defaults']['mean_test'] - all_results['stage2_tuned']['mean_test']) 
                      / all_results['stage1_defaults']['mean_test'] * 100)
        f.write(f"  Improvement: {improvement:.2f}%\n\n")
        
        f.write("Stage 3a: Random Optimization\n")
        f.write(f"  Best Algorithm: {all_results['stage3_part1_ro']['best_algo']}\n")
        f.write(f"  Test Loss: {all_results['stage3_part1_ro']['mean_test']:.6f} ± {all_results['stage3_part1_ro']['std_test']:.6f}\n")
        f.write("  All Algorithms:\n")
        for algo, result in all_results['stage3_part1_ro']['all_algos'].items():
            f.write(f"    {algo}: {result['mean_test']:.6f} ± {result['std_test']:.6f}\n")
        f.write("\n")
        
        f.write("Stage 3b: Adam Ablations\n")
        f.write(f"  Best Optimizer: {all_results['stage3_part2_adam']['best_optimizer']}\n")
        f.write(f"  Test Loss: {all_results['stage3_part2_adam']['mean_test']:.6f} ± {all_results['stage3_part2_adam']['std_test']:.6f}\n")
        f.write("  All Optimizers:\n")
        for opt, result in all_results['stage3_part2_adam']['all_optimizers'].items():
            f.write(f"    {opt}: {result['mean_test']:.6f} ± {result['std_test']:.6f}\n")
        f.write("\n")
        
        f.write("Stage 3c: Targeted Regularization\n")
        f.write(f"  Best Config: {all_results['stage3_part3_reg']['best_config']}\n")
        f.write(f"  Test Loss: {all_results['stage3_part3_reg']['mean_test']:.6f} ± {all_results['stage3_part3_reg']['std_test']:.6f}\n")
        f.write("  All Configs:\n")
        for cfg, result in all_results['stage3_part3_reg']['all_configs'].items():
            f.write(f"    {cfg}: {result['mean_test']:.6f} ± {result['std_test']:.6f}\n")
        f.write("\n")
        
        f.write("="*100 + "\n")
        f.write(f"JSON results: {json_path}\n")
        f.write("="*100 + "\n")
    
    print(f"\n{'='*100}")
    print(f"Comparison report saved to: {report_path}")
    print(f"JSON results saved to: {json_path}")
    print(f"{'='*100}\n")
    
    return all_results


if __name__ == '__main__':
    main_comparison()

