"""unified plotting module for rl report

all plotting functionality consolidated:
- publication-quality report plots (aggregated across seeds)
- policy visualization
- comparative analysis

usage:
    python src/utils/generate_report_plots.py
"""
import sys
from pathlib import Path

# add src to path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
import gymnasium as gym
from environments.cartpole_discretizer import StateDiscretizer

# publication-quality settings
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,  # high-res for report
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
})

# consistent algorithm colors across all plots
ALGO_COLORS = {
    'sarsa': '#E24A33',     # red
    'qlearning': '#348ABD', # blue
    'vi': '#2CA02C',        # green
    'pi': '#9467BD',        # purple
}


def load_master_summary(algo: str, results_dir: Path) -> Dict:
    """load master summary json for algorithm
    
    args:
        algo: algorithm name (sarsa, qlearning, vi, pi)
        results_dir: path to results directory (containing raw/ subdirectory)
    """
    json_path = results_dir / 'raw' / algo / 'master_summary.json'
    if json_path.exists():
        with open(json_path, 'r') as f:
            return json.load(f)
    print(f"warning: master summary not found at {json_path}")
    return {}


# =============================================================================
# REPORT PLOTS: MULTI-SEED AGGREGATED VISUALIZATIONS (4 figures total)
# =============================================================================



def plot_learning_curves_model_free(results_dir: Path, output_dir: Path):
    """Plot 1: Model-Free Learning Curves (SARSA vs Q-Learning)
    
    1x2 grid showing rolling average return for both environments.
    VI/PI optimal performance shown as reference lines.
    Note: For CartPole, reward = episode length (mentioned in report).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # VI/PI optimal returns (from master_summary.json)
    vi_pi_optimal = {
        'blackjack': -0.038,  # both VI and PI converge to same optimal
        'cartpole': 500       # max episode length (optimal policy)
    }
    
    for col, env in enumerate(['blackjack', 'cartpole']):
        ax = axes[col]
        window = 500 if env == 'blackjack' else 100
        
        for algo in ['sarsa', 'qlearning']:
            csv_dir = results_dir / 'raw' / algo / env
            csv_files = list(csv_dir.glob(f'{algo}_{env}_seed*.csv'))
            
            if not csv_files:
                continue
            
            all_rolling = []
            for f in csv_files:
                df = pd.read_csv(f)
                if 'episode_return' in df.columns:
                    returns = df['episode_return'].values
                    rolling = pd.Series(returns).rolling(window=window, min_periods=1).mean().values
                    all_rolling.append(rolling)
            
            if not all_rolling:
                continue
            
            min_len = min(len(r) for r in all_rolling)
            rolling_array = np.array([r[:min_len] for r in all_rolling])
            
            mean = rolling_array.mean(axis=0)
            q1 = np.percentile(rolling_array, 25, axis=0)
            q3 = np.percentile(rolling_array, 75, axis=0)
            
            episodes = np.arange(len(mean))
            ax.plot(episodes, mean, label=algo.upper(), color=ALGO_COLORS[algo], linewidth=2)
            ax.fill_between(episodes, q1, q3, color=ALGO_COLORS[algo], alpha=0.2)
        
        # VI/PI optimal performance reference lines
        if env in vi_pi_optimal:
            opt_val = vi_pi_optimal[env]
            ax.axhline(y=opt_val, color=ALGO_COLORS['vi'], linestyle='--', linewidth=2,
                      alpha=0.8, label=f'VI ({opt_val:.2f})')
            ax.axhline(y=opt_val, color=ALGO_COLORS['pi'], linestyle=':', linewidth=2,
                      alpha=0.8, label=f'PI ({opt_val:.2f})')
        
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_title(f'{env.title()}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if env == 'blackjack':
            ax.set_ylabel(f'Rolling Avg Return (w={window})', fontsize=11)
            ax.set_ylim(-0.5, 0.1)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4, linewidth=1, label='_Break-even')
        else:
            ax.set_ylabel(f'Episode Length / Return (w={window})', fontsize=11)
            ax.set_ylim(0, 550)
            ax.axhline(y=195, color='gray', linestyle='--', alpha=0.4, linewidth=1, label='_Solved threshold')
    
    plt.suptitle('Model-Free RL: SARSA vs Q-Learning\n(Mean ± IQR across 40 seeds) · VI/PI optimal shown as reference', 
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'model_free_learning_curves.png'
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"saved: {output_path}")


def plot_convergence_model_based(results_dir: Path, output_dir: Path):
    """Model-Based Convergence: VI vs PI comparison
    
    Clean, minimal visualization showing the key insight:
    - VI: many incremental updates (slow but steady)
    - PI: few big jumps (fast outer loop)
    
    Single figure with 2 rows x 2 columns:
    - Columns: Blackjack | CartPole
    - Row 1: VI delta (log scale, thin line)
    - Row 2: PI iterations bar + total work comparison
    """
    fig = plt.figure(figsize=(12, 5))
    
    # create gridspec for flexible layout
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
    
    # colors
    vi_color = '#2196F3'  # blue
    pi_color = '#FF9800'  # orange
    
    envs = ['blackjack', 'cartpole']
    env_labels = ['Blackjack', 'CartPole']
    
    # collect data for summary
    summary_data = {}
    
    for col, (env, label) in enumerate(zip(envs, env_labels)):
        ax = fig.add_subplot(gs[col])
        
        # --- VI data ---
        vi_csv_dir = results_dir / 'raw' / 'vi' / env
        vi_files = list(vi_csv_dir.glob(f'vi_{env}_seed*.csv'))
        
        vi_iters = 0
        vi_deltas = []
        if vi_files:
            for f in vi_files:
                df = pd.read_csv(f)
                if 'delta' in df.columns:
                    vi_deltas.append(df['delta'].values)
            if vi_deltas:
                vi_iters = int(np.mean([len(d) for d in vi_deltas]))
        
        # --- PI data ---
        pi_csv_dir = results_dir / 'raw' / 'pi' / env
        pi_files = list(pi_csv_dir.glob(f'pi_{env}_seed*.csv'))
        
        pi_outer_iters = 0
        pi_total_evals = 0
        if pi_files:
            for f in pi_files:
                df = pd.read_csv(f)
                if 'policy_changes' in df.columns:
                    pi_outer_iters = max(pi_outer_iters, len(df))
                if 'eval_iterations' in df.columns:
                    pi_total_evals = max(pi_total_evals, df['eval_iterations'].sum())
        
        summary_data[env] = {
            'vi_iters': vi_iters,
            'pi_outer': pi_outer_iters,
            'pi_total_evals': int(pi_total_evals)
        }
        
        # --- plot: horizontal bar comparison ---
        y_pos = np.array([0, 1])
        bar_height = 0.5
        
        # normalize to show relative iterations
        max_iters = max(vi_iters, pi_outer_iters + pi_total_evals)
        
        # VI: single bar (iterations = total work)
        ax.barh(y_pos[0], vi_iters, height=bar_height, color=vi_color, alpha=0.8,
               label=f'VI: {vi_iters} iterations')
        
        # PI: stacked bar (outer + inner evals)  
        ax.barh(y_pos[1], pi_outer_iters, height=bar_height, color=pi_color, alpha=0.8,
               label=f'PI outer: {pi_outer_iters}')
        ax.barh(y_pos[1], pi_total_evals, height=bar_height, left=pi_outer_iters,
               color=pi_color, alpha=0.4, label=f'PI eval sweeps: {int(pi_total_evals)}')
        
        # annotations
        ax.text(vi_iters + max_iters*0.02, y_pos[0], f'{vi_iters}', 
               va='center', fontsize=11, fontweight='bold')
        ax.text(pi_outer_iters + pi_total_evals + max_iters*0.02, y_pos[1], 
               f'{pi_outer_iters} + {int(pi_total_evals)}', 
               va='center', fontsize=10, color='#666')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(['Value\nIteration', 'Policy\nIteration'], fontsize=11)
        ax.set_xlabel('Total Iterations / Sweeps', fontsize=11)
        ax.set_title(f'{label}', fontsize=13, fontweight='bold')
        ax.set_xlim(0, max_iters * 1.25)
        ax.grid(True, alpha=0.3, axis='x')
        
        # remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle('VI vs PI: Convergence Comparison\n(VI: many small steps | PI: few outer loops + evaluation sweeps)', 
                fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'model_based_convergence.png'
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"saved: {output_path}")


def plot_final_performance_comparison(results_dir: Path, output_dir: Path):
    """Plot 3: Final Performance Summary (All 4 Algorithms)
    
    Bar chart comparing all algorithms on both environments:
    - VI, PI: Policy evaluation return
    - SARSA, Q-Learning: Final 10% episode return
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {
        'vi': "#A861AC",
        'pi': '#2E8B57', 
        'sarsa': '#E24A33',
        'qlearning': '#348ABD'
    }
    
    algo_labels = ['VI', 'PI', 'SARSA', 'Q-Learning']
    algo_keys = ['vi', 'pi', 'sarsa', 'qlearning']
    
    for col, env in enumerate(['blackjack', 'cartpole']):
        ax = axes[col]
        
        means = []
        stds = []
        valid_algos = []
        valid_colors = []
        
        for algo in algo_keys:
            csv_dir = results_dir / 'raw' / algo / env
            
            if algo in ['vi', 'pi']:
                # model-based: get policy eval return from JSON
                json_files = list(csv_dir.glob(f'{algo}_{env}_seed*.json'))
                if json_files:
                    returns = []
                    for jf in json_files:
                        with open(jf) as f:
                            data = json.load(f)
                        if 'results' in data and 'policy_evaluation' in data['results']:
                            returns.append(data['results']['policy_evaluation']['mean_return'])
                    if returns:
                        means.append(np.mean(returns))
                        stds.append(np.std(returns))
                        valid_algos.append(algo_labels[algo_keys.index(algo)])
                        valid_colors.append(colors[algo])
            else:
                # model-free: get final 10% episode return from CSV
                csv_files = list(csv_dir.glob(f'{algo}_{env}_seed*.csv'))
                if csv_files:
                    final_returns = []
                    for f in csv_files:
                        df = pd.read_csv(f)
                        if 'episode_return' in df.columns:
                            returns = df['episode_return'].values
                            final_chunk = returns[int(len(returns)*0.9):]
                            final_returns.append(np.mean(final_chunk))
                    if final_returns:
                        means.append(np.mean(final_returns))
                        stds.append(np.std(final_returns))
                        valid_algos.append(algo_labels[algo_keys.index(algo)])
                        valid_colors.append(colors[algo])
        
        if means:
            x = np.arange(len(means))
            bars = ax.bar(x, means, yerr=stds, capsize=5, color=valid_colors, alpha=0.8, edgecolor='black')
            
            # add value labels on bars
            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                ax.annotate(f'{mean:.2f}',
                            xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=(5, 0), textcoords='offset points',
                            ha='left', va='center', fontsize=10, fontweight='bold')
            
            ax.set_xticks(x)
            ax.set_xticklabels(valid_algos, fontsize=11)
            ax.set_ylabel('Mean Return', fontsize=11)
            ax.set_title(f'{env.title()}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # reference lines
            if env == 'blackjack':
                ax.axhline(y=0, color='green', linestyle='--', alpha=0.6, linewidth=1.5, label='Break-even (0)')
                ax.set_ylim(-0.25, 0.1)
            else:
                ax.axhline(y=195, color='green', linestyle='--', alpha=0.6, linewidth=1.5, label='Solved threshold (195)')
                # auto-scale y-axis to fit all bars + error bars with 10% padding
                max_height = max(m + s for m, s in zip(means, stds))
                ax.set_ylim(0, max_height * 1.15)
            # legend entry for the reference line to explain green dotted key
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(handles[-1:], labels[-1:], loc='upper left', fontsize=9)
    
    plt.suptitle('Final Performance Comparison: All Algorithms\n(Mean ± Std across seeds)', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    output_path = output_dir / 'final_performance_comparison.png'
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"saved: {output_path}")


def plot_blackjack_policy_heatmap(results_dir: Path, output_dir: Path):
    """blackjack combined policy & value visualization with diagonal split cells
    
    generates 4 heatmaps (VI, PI, SARSA, Q-Learning) in a 2x2 grid
    each cell is split diagonally:
    - upper-left triangle: no usable ace
    - lower-right triangle: usable ace
    color: green (high value) to red (low value)
    text: S = STICK, H = HIT
    """
    
    # state space definition: player_sum 12-21, dealer 1-10, usable_ace (200 states)
    def build_state_space():
        states = []
        for player_sum in range(12, 22):  # 12-21
            for dealer_card in range(1, 11):  # 1(ace)-10
                for usable_ace in [False, True]:
                    states.append((player_sum, dealer_card, usable_ace))
        return states
    
    states = build_state_space()
    state_to_idx = {s: i for i, s in enumerate(states)}
    
    player_sums = list(range(12, 22))  # 12-21 (y-axis, rows)
    dealer_showing = list(range(1, 11))  # 1(ace)-10 (x-axis, cols)
    
    # algorithm configs: (algo_key, display_name, has_value_file)
    algos = [
        ('vi', 'Value Iteration (VI)', True),
        ('pi', 'Policy Iteration (PI)', True),
        ('sarsa', 'SARSA (On-Policy)', False),
        ('qlearning', 'Q-Learning (Off-Policy)', False),
    ]
    
    # === 2x2 GRID: Policy (S/H) with Value (color) ===
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()
    plt.subplots_adjust(wspace=0.12, hspace=0.22)
    
    # get value ranges for colormap normalization (from all algorithms)
    all_values = []
    for algo, _, has_value_file in algos:
        if has_value_file:
            value_dir = results_dir / 'raw' / algo / 'blackjack'
            value_files = sorted(value_dir.glob(f'{algo}_blackjack_seed*_value.npy'))
            if value_files:
                value = np.load(value_files[0], allow_pickle=True)
                all_values.extend(value.flatten())
        else:
            # extract max Q-values from Q-table for model-free algorithms
            qtable_dir = results_dir / 'raw' / algo / 'blackjack'
            qtable_files = sorted(qtable_dir.glob(f'{algo}_blackjack_seed*_qtable.npy'))
            if qtable_files:
                qtable = np.load(qtable_files[0], allow_pickle=True).item()
                # get max Q(s,a) for each state
                state_values = {}
                for (state, action), q_val in qtable.items():
                    if state not in state_values:
                        state_values[state] = q_val
                    else:
                        state_values[state] = max(state_values[state], q_val)
                all_values.extend(state_values.values())
    
    vmin, vmax = min(all_values), max(all_values)
    
    # create green-red colormap (red = low/bad, green = high/good)
    cmap = plt.cm.RdYlGn  # red -> yellow -> green
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    for idx, (algo, algo_name, has_value_file) in enumerate(algos):
        ax = axes[idx]
        
        # load policy and values based on algorithm type
        data_dir = results_dir / 'raw' / algo / 'blackjack'
        policy_files = sorted(data_dir.glob(f'{algo}_blackjack_seed*_policy.npy'))
        
        if not policy_files:
            print(f"warning: missing policy files for {algo}")
            continue
        
        policy_data = np.load(policy_files[0], allow_pickle=True)
        
        # build lookups based on data structure
        policy_no_ace = np.zeros((len(player_sums), len(dealer_showing)))
        policy_usable_ace = np.zeros((len(player_sums), len(dealer_showing)))
        value_no_ace = np.zeros((len(player_sums), len(dealer_showing)))
        value_usable_ace = np.zeros((len(player_sums), len(dealer_showing)))
        
        if has_value_file:
            # VI/PI: policy and value are arrays indexed by state_to_idx
            value_files = sorted(data_dir.glob(f'{algo}_blackjack_seed*_value.npy'))
            value_data = np.load(value_files[0], allow_pickle=True)
            
            for i, ps in enumerate(player_sums):
                for j, ds in enumerate(dealer_showing):
                    state_no_ace = (ps, ds, False)
                    state_ace = (ps, ds, True)
                    if state_no_ace in state_to_idx:
                        policy_no_ace[i, j] = policy_data[state_to_idx[state_no_ace]]
                        value_no_ace[i, j] = value_data[state_to_idx[state_no_ace]]
                    if state_ace in state_to_idx:
                        policy_usable_ace[i, j] = policy_data[state_to_idx[state_ace]]
                        value_usable_ace[i, j] = value_data[state_to_idx[state_ace]]
        else:
            # SARSA/Q-Learning: policy and qtable are dicts with (state) or ((state), action) keys
            policy_dict = policy_data.item() if policy_data.shape == () else policy_data
            
            qtable_files = sorted(data_dir.glob(f'{algo}_blackjack_seed*_qtable.npy'))
            qtable_data = np.load(qtable_files[0], allow_pickle=True)
            qtable_dict = qtable_data.item() if qtable_data.shape == () else qtable_data
            
            # extract state values as max Q(s,a) over actions
            state_values = {}
            for (state, action), q_val in qtable_dict.items():
                if state not in state_values:
                    state_values[state] = q_val
                else:
                    state_values[state] = max(state_values[state], q_val)
            
            for i, ps in enumerate(player_sums):
                for j, ds in enumerate(dealer_showing):
                    # state format: (player_sum, dealer_card, usable_ace as 0/1)
                    state_no_ace = (ps, ds, 0)
                    state_ace = (ps, ds, 1)
                    
                    if state_no_ace in policy_dict:
                        policy_no_ace[i, j] = policy_dict[state_no_ace]
                    if state_ace in policy_dict:
                        policy_usable_ace[i, j] = policy_dict[state_ace]
                    if state_no_ace in state_values:
                        value_no_ace[i, j] = state_values[state_no_ace]
                    if state_ace in state_values:
                        value_usable_ace[i, j] = state_values[state_ace]
        
        # draw each cell with diagonal split - color from value, text from policy
        for i, ps in enumerate(player_sums):
            for j, ds in enumerate(dealer_showing):
                x0, x1 = j - 0.5, j + 0.5
                y0, y1 = i - 0.5, i + 0.5
                
                # upper-left triangle (no usable ace) - color by value
                val_no_ace = value_no_ace[i, j]
                color_no_ace = cmap(norm(val_no_ace))
                tri_upper = Polygon([(x0, y0), (x1, y0), (x0, y1)], closed=True,
                                   facecolor=color_no_ace, edgecolor='white', linewidth=0.5)
                ax.add_patch(tri_upper)
                
                # lower-right triangle (usable ace) - color by value
                val_ace = value_usable_ace[i, j]
                color_ace = cmap(norm(val_ace))
                tri_lower = Polygon([(x1, y0), (x1, y1), (x0, y1)], closed=True,
                                   facecolor=color_ace, edgecolor='white', linewidth=0.5)
                ax.add_patch(tri_lower)
                
                # text labels - S/H from policy (always black)
                action_no_ace = int(policy_no_ace[i, j])
                action_ace = int(policy_usable_ace[i, j])
                text_no_ace = 'H' if action_no_ace == 1 else 'S'
                text_ace = 'H' if action_ace == 1 else 'S'
                
                ax.text(j - 0.22, i - 0.22, text_no_ace, ha='center', va='center',
                       fontsize=9, fontweight='bold', color='black')
                ax.text(j + 0.22, i + 0.22, text_ace, ha='center', va='center',
                       fontsize=9, fontweight='bold', color='black')
        
        # set axis limits and appearance
        ax.set_xlim(-0.5, len(dealer_showing) - 0.5)
        ax.set_ylim(len(player_sums) - 0.5, -0.5)
        
        ax.set_xticks(range(len(dealer_showing)))
        ax.set_xticklabels(['A'] + list(range(2, 11)), fontsize=12)
        ax.set_xlabel('Dealer Showing', fontsize=13, fontweight='bold')
        
        ax.set_yticks(range(len(player_sums)))
        ax.set_yticklabels(player_sums, fontsize=11)
        ax.set_ylabel('Player Sum', fontsize=13, fontweight='bold')
        
        ax.set_title(algo_name, fontsize=15, fontweight='bold', pad=10)
        
        for x in np.arange(-0.5, len(dealer_showing), 1):
            ax.axvline(x, color='gray', linewidth=0.5, alpha=0.5)
        for y in np.arange(-0.5, len(player_sums), 1):
            ax.axhline(y, color='gray', linewidth=0.5, alpha=0.5)
        
        ax.set_aspect('equal')
    
    # vertical colorbar on the right (aligned with plot height)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.93, 0.08, 0.02, 0.84])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('State Value V(s)\n(Expected Return: Win=+1, Lose=-1)', fontsize=12)
    
    # main title (bold) and subtitle key (not bold) - centered
    fig.text(0.52, 0.97, 'Blackjack Optimal Policy & State Value Heatmaps',
            ha='center', fontsize=20, fontweight='bold')
    fig.text(0.52, 0.955, 'Upper-left ◸ = No Usable Ace  |  Lower-right ◿ = Usable Ace  |  S = STICK, H = HIT  |  Color = State Value',
            ha='center', fontsize=12, style='italic')
    
    plt.subplots_adjust(top=0.92)
    output_path = output_dir / 'blackjack_heatmap.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {output_path}")


def plot_cartpole_episode_length(results_dir: Path, output_dir: Path):
    """CartPole episode length: SARSA vs Q-Learning overlaid on single plot
    
    Shows balancing performance over training with both algorithms for direct comparison.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    window = 100  # rolling average window size
    colors = {'sarsa': '#E24A33', 'qlearning': '#348ABD'}
    
    for algo in ['sarsa', 'qlearning']:
        csv_dir = results_dir / 'raw' / algo / 'cartpole'
        csv_files = list(csv_dir.glob(f'{algo}_cartpole_seed*.csv'))
        
        if not csv_files:
            continue
        
        all_rolling = []
        for f in csv_files:
            df = pd.read_csv(f)
            if 'steps' in df.columns:
                steps = df['steps'].values
                rolling_avg = pd.Series(steps).rolling(window=window, min_periods=1).mean().values
                all_rolling.append(rolling_avg)
        
        if not all_rolling:
            continue
        
        min_len = min(len(r) for r in all_rolling)
        rolling_array = np.array([r[:min_len] for r in all_rolling])
        
        mean = rolling_array.mean(axis=0)
        q1 = np.percentile(rolling_array, 25, axis=0)
        q3 = np.percentile(rolling_array, 75, axis=0)
        
        episodes = np.arange(min_len)
        
        ax.plot(episodes, mean, color=colors[algo], linewidth=2.5, 
               label=f'{algo.upper()}', alpha=0.9)
        ax.fill_between(episodes, q1, q3, color=colors[algo], alpha=0.15)
    
    # reference lines
    ax.axhline(y=500, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='Max (500)')
    ax.axhline(y=195, color='green', linestyle=':', alpha=0.7, linewidth=2, label='Solved (195)')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Steps per Episode (100-ep Rolling Avg)', fontsize=12)
    ax.set_title('CartPole Balancing: SARSA vs Q-Learning\n(Mean ± IQR across 40 seeds)', 
                fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 550)
    ax.set_xlim(0, min_len)
    
    output_path = output_dir / 'cartpole_episode_length.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {output_path}")


def plot_vi_pi_convergence_curves(results_dir: Path, output_dir: Path):
    """vi vs pi convergence curves - focused on blackjack where scales are comparable
    
    answers: how many iterations for convergence? which is faster? why?
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # === LEFT: Blackjack - VI delta decay ===
    ax1 = axes[0]
    
    vi_csv_dir = results_dir / 'raw' / 'vi' / 'blackjack'
    vi_csv_files = sorted(vi_csv_dir.glob('vi_blackjack_seed*.csv'))
    
    vi_deltas = []
    for f in vi_csv_files:
        try:
            df = pd.read_csv(f)
            if 'delta' in df.columns:
                vi_deltas.append(df['delta'].values)
        except:
            continue
    
    if vi_deltas:
        min_len = min(len(d) for d in vi_deltas)
        vi_arr = np.array([d[:min_len] for d in vi_deltas])
        vi_mean = vi_arr.mean(axis=0)
        vi_q1 = np.percentile(vi_arr, 25, axis=0)
        vi_q3 = np.percentile(vi_arr, 75, axis=0)
        iters = np.arange(len(vi_mean))
        
        ax1.plot(iters, vi_mean, 'o-', color='#2196F3', linewidth=2.5, 
                markersize=8, label='VI: Bellman error (δ)')
        ax1.fill_between(iters, vi_q1, vi_q3, color='#2196F3', alpha=0.2)
        
        # mark convergence
        threshold = 0.0001
        conv_idx = np.where(vi_mean < threshold)[0]
        if len(conv_idx) > 0:
            ax1.axvline(x=conv_idx[0], color='#2196F3', linestyle='--', alpha=0.8,
                       linewidth=2, label=f'Converged: iter {conv_idx[0]}')
            ax1.plot(conv_idx[0], vi_mean[conv_idx[0]], 'o', color='red', 
                    markersize=12, zorder=10, markeredgecolor='white', markeredgewidth=2)
    
    ax1.axhline(y=0.0001, color='red', linestyle=':', alpha=0.6, linewidth=2, 
               label='θ = 0.0001')
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Max Value Change (δ)', fontsize=12, fontweight='bold')
    ax1.set_title('Value Iteration: Blackjack\nBellman Error Decay', fontsize=13, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    
    # === RIGHT: Blackjack - PI policy changes ===
    ax2 = axes[1]
    
    pi_csv_dir = results_dir / 'raw' / 'pi' / 'blackjack'
    pi_csv_files = sorted(pi_csv_dir.glob('pi_blackjack_seed*.csv'))
    
    pi_changes = []
    pi_eval_iters = []
    for f in pi_csv_files:
        try:
            df = pd.read_csv(f)
            if 'policy_changes' in df.columns:
                pi_changes.append(df['policy_changes'].values)
            if 'eval_iterations' in df.columns:
                pi_eval_iters.append(df['eval_iterations'].values)
        except:
            continue
    
    if pi_changes:
        min_len = min(len(c) for c in pi_changes)
        pi_arr = np.array([c[:min_len] for c in pi_changes])
        pi_mean = pi_arr.mean(axis=0)
        iters = np.arange(len(pi_mean))
        
        ax2.plot(iters, pi_mean, 's-', color='#FF9800', linewidth=2.5,
                markersize=10, label='Policy changes per iteration')
        
        # mark convergence (when changes = 0)
        conv_idx = np.where(pi_mean == 0)[0]
        if len(conv_idx) > 0:
            ax2.axvline(x=conv_idx[0], color='#FF9800', linestyle='--', alpha=0.8,
                       linewidth=2, label=f'Converged: iter {conv_idx[0]}')
            ax2.plot(conv_idx[0], pi_mean[conv_idx[0]], 's', color='red',
                    markersize=12, zorder=10, markeredgecolor='white', markeredgewidth=2)
    
    ax2.axhline(y=0, color='red', linestyle=':', alpha=0.6, linewidth=2,
               label='Stable policy (0 changes)')
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Actions Changed', fontsize=12, fontweight='bold')
    ax2.set_title('Policy Iteration: Blackjack\nPolicy Improvement Steps', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Blackjack: VI vs PI Convergence Dynamics', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = output_dir / 'vi_pi_convergence_curves.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {output_path}")


def plot_cartpole_convergence_curves(results_dir: Path, output_dir: Path):
    """cartpole vi vs pi convergence - separate plot due to scale differences
    
    shows the key insight: PI converges in few outer iterations but each is expensive
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # === LEFT: VI delta decay (log scale, many iterations) ===
    ax1 = axes[0]
    
    vi_csv_dir = results_dir / 'raw' / 'vi' / 'cartpole'
    vi_csv_files = sorted(vi_csv_dir.glob('vi_cartpole_seed*.csv'))
    
    vi_deltas = []
    for f in vi_csv_files:
        try:
            df = pd.read_csv(f)
            if 'delta' in df.columns:
                vi_deltas.append(df['delta'].values)
        except:
            continue
    
    if vi_deltas:
        min_len = min(len(d) for d in vi_deltas)
        vi_arr = np.array([d[:min_len] for d in vi_deltas])
        vi_mean = vi_arr.mean(axis=0)
        vi_q1 = np.percentile(vi_arr, 25, axis=0)
        vi_q3 = np.percentile(vi_arr, 75, axis=0)
        iters = np.arange(len(vi_mean))
        
        # subsample for cleaner plot (every 50th point)
        step = max(1, len(iters) // 20)
        ax1.plot(iters[::step], vi_mean[::step], 'o-', color='#2196F3', linewidth=2, 
                markersize=6, label=f'VI: {len(vi_mean)} iterations')
        ax1.fill_between(iters, vi_q1, vi_q3, color='#2196F3', alpha=0.15)
        
        # mark convergence
        threshold = 0.0001
        conv_idx = np.where(vi_mean < threshold)[0]
        if len(conv_idx) > 0:
            ax1.plot(conv_idx[0], vi_mean[conv_idx[0]], 'o', color='red', 
                    markersize=14, zorder=10, markeredgecolor='white', markeredgewidth=2,
                    label=f'Converged: iter {conv_idx[0]}')
    
    ax1.axhline(y=0.0001, color='red', linestyle=':', alpha=0.6, linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Max Value Change (δ)', fontsize=12, fontweight='bold')
    ax1.set_title('Value Iteration: CartPole\nSlow Convergence (918 iterations)', 
                 fontsize=13, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    
    # === RIGHT: PI outer loop + inner loop breakdown ===
    ax2 = axes[1]
    
    pi_csv_dir = results_dir / 'raw' / 'pi' / 'cartpole'
    pi_csv_files = sorted(pi_csv_dir.glob('pi_cartpole_seed*.csv'))
    
    pi_changes = []
    pi_eval_iters = []
    for f in pi_csv_files:
        try:
            df = pd.read_csv(f)
            if 'policy_changes' in df.columns:
                pi_changes.append(df['policy_changes'].values)
            if 'eval_iterations' in df.columns:
                pi_eval_iters.append(df['eval_iterations'].values)
        except:
            continue
    
    if pi_changes and pi_eval_iters:
        min_len = min(len(c) for c in pi_changes)
        pi_arr = np.array([c[:min_len] for c in pi_changes])
        pi_mean = pi_arr.mean(axis=0)
        
        min_len_eval = min(len(e) for e in pi_eval_iters)
        eval_arr = np.array([e[:min_len_eval] for e in pi_eval_iters])
        eval_mean = eval_arr.mean(axis=0)
        
        iters = np.arange(len(pi_mean))
        
        # plot policy changes (left y-axis)
        line1 = ax2.plot(iters, pi_mean, 's-', color='#FF9800', linewidth=2.5,
                        markersize=12, label='Policy changes')
        ax2.set_ylabel('Actions Changed', fontsize=12, fontweight='bold', color='#FF9800')
        ax2.tick_params(axis='y', labelcolor='#FF9800')
        
        # plot eval iterations (right y-axis)
        ax2b = ax2.twinx()
        line2 = ax2b.plot(iters[:len(eval_mean)], eval_mean, 'D--', color='#4CAF50', 
                         linewidth=2, markersize=10, label='Eval sweeps per iteration')
        ax2b.set_ylabel('Policy Eval Sweeps', fontsize=12, fontweight='bold', color='#4CAF50')
        ax2b.tick_params(axis='y', labelcolor='#4CAF50')
        
        # combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper right', fontsize=10)
        
        # add annotation for total work
        total_eval = eval_mean.sum()
        ax2.annotate(f'Total eval sweeps: {total_eval:.0f}', 
                    xy=(0.5, 0.95), xycoords='axes fraction',
                    fontsize=11, fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.set_xlabel('Outer Iteration (Policy Improvement)', fontsize=12, fontweight='bold')
    ax2.set_title('Policy Iteration: CartPole\nFast Outer Loop (5 iterations)', 
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('CartPole: VI vs PI Convergence\n(VI: many small steps vs PI: few large steps)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = output_dir / 'cartpole_convergence_curves.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {output_path}")


# =============================================================================
# OPTIONAL PLOTS: Additional visualizations (not required, but informative)
# =============================================================================

def plot_exploration_decay(results_dir: Path, output_dir: Path):
    """Optional: Exploration rate (ε) decay over training
    
    Shows how exploration decreases over episodes for all algorithms.
    VI/PI shown as horizontal lines at ε=0 (deterministic from model).
    1x2 grid: Blackjack | CartPole
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for col, env in enumerate(['blackjack', 'cartpole']):
        ax = axes[col]
        
        # model-free algorithms (episode-based curves)
        for algo in ['sarsa', 'qlearning']:
            csv_dir = results_dir / 'raw' / algo / env
            csv_files = list(csv_dir.glob(f'{algo}_{env}_seed*.csv'))
            
            if not csv_files:
                continue
            
            all_epsilon = []
            for f in csv_files:
                df = pd.read_csv(f)
                if 'exploration_ratio' in df.columns:
                    epsilon = df['exploration_ratio'].values
                    all_epsilon.append(epsilon)
            
            if not all_epsilon:
                continue
            
            min_len = min(len(e) for e in all_epsilon)
            epsilon_array = np.array([e[:min_len] for e in all_epsilon])
            
            mean = epsilon_array.mean(axis=0)
            q1 = np.percentile(epsilon_array, 25, axis=0)
            q3 = np.percentile(epsilon_array, 75, axis=0)
            
            episodes = np.arange(len(mean))
            ax.plot(episodes, mean, label=algo.upper(), color=ALGO_COLORS[algo], linewidth=2)
            ax.fill_between(episodes, q1, q3, color=ALGO_COLORS[algo], alpha=0.2)
        
        # VI and PI: deterministic (ε=0) - shown as horizontal lines
        ax.axhline(y=0, color=ALGO_COLORS['vi'], linestyle='--', linewidth=2, 
                   alpha=0.8, label='VI (ε=0)')
        ax.axhline(y=0, color=ALGO_COLORS['pi'], linestyle=':', linewidth=2,
                   alpha=0.8, label='PI (ε=0)')
        
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Exploration Rate (ε)', fontsize=11)
        ax.set_title(f'{env.title()}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    
    plt.suptitle('Exploration Rate Decay (ε-greedy)\n(Mean ± IQR across seeds) · VI/PI are deterministic (ε=0)', 
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'optional' / 'exploration_decay.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"saved: {output_path}")


def plot_td_error_decay(results_dir: Path, output_dir: Path):
    """Optional: TD Error / Bellman Error decay over training
    
    Shows Q-value convergence via mean absolute TD error (model-free)
    and Bellman error delta (model-based).
    2x2 grid: Model-Free (top) | Model-Based (bottom) × Blackjack | CartPole
    """
    fig = plt.figure(figsize=(12, 10))
    
    # === TOP ROW: Model-Free TD Error ===
    for col, env in enumerate(['blackjack', 'cartpole']):
        ax = fig.add_subplot(2, 2, col + 1)
        window = 500 if env == 'blackjack' else 100
        
        for algo in ['sarsa', 'qlearning']:
            csv_dir = results_dir / 'raw' / algo / env
            csv_files = list(csv_dir.glob(f'{algo}_{env}_seed*.csv'))
            
            if not csv_files:
                continue
            
            all_td = []
            for f in csv_files:
                df = pd.read_csv(f)
                if 'mean_abs_td_error' in df.columns:
                    td_error = df['mean_abs_td_error'].values
                    rolling = pd.Series(td_error).rolling(window=window, min_periods=1).mean().values
                    all_td.append(rolling)
            
            if not all_td:
                continue
            
            min_len = min(len(t) for t in all_td)
            td_array = np.array([t[:min_len] for t in all_td])
            
            mean = td_array.mean(axis=0)
            q1 = np.percentile(td_array, 25, axis=0)
            q3 = np.percentile(td_array, 75, axis=0)
            
            episodes = np.arange(len(mean))
            ax.plot(episodes, mean, label=algo.upper(), color=ALGO_COLORS[algo], linewidth=2)
            ax.fill_between(episodes, q1, q3, color=ALGO_COLORS[algo], alpha=0.2)
        
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel(f'Mean |TD Error| (w={window})', fontsize=11)
        ax.set_title(f'{env.title()} - Model-Free', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # === BOTTOM ROW: Model-Based Bellman Error (delta) ===
    for col, env in enumerate(['blackjack', 'cartpole']):
        ax = fig.add_subplot(2, 2, col + 3)
        
        # VI: delta decay
        csv_dir = results_dir / 'raw' / 'vi' / env
        csv_files = list(csv_dir.glob('*.csv'))
        if csv_files:
            df = pd.read_csv(csv_files[0])
            if 'delta' in df.columns:
                ax.semilogy(df['iteration'], df['delta'], '-o', color=ALGO_COLORS['vi'], 
                           linewidth=2, markersize=4, label='VI (Δ)')
        
        # PI: we can plot eval_iterations as a proxy for convergence work
        csv_dir = results_dir / 'raw' / 'pi' / env
        csv_files = list(csv_dir.glob('*.csv'))
        if csv_files:
            df = pd.read_csv(csv_files[0])
            if 'policy_changes' in df.columns:
                # normalize policy changes to show decay pattern
                ax2 = ax.twinx()
                ax2.plot(df['iteration'], df['policy_changes'], '-s', color=ALGO_COLORS['pi'],
                        linewidth=2, markersize=6, label='PI (policy Δ)')
                ax2.set_ylabel('Policy Changes', fontsize=11, color=ALGO_COLORS['pi'])
                ax2.tick_params(axis='y', labelcolor=ALGO_COLORS['pi'])
        
        ax.axhline(y=0.0001, color='gray', linestyle=':', alpha=0.7, linewidth=1.5, label='θ=0.0001')
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Bellman Error (Δ)', fontsize=11, color=ALGO_COLORS['vi'])
        ax.tick_params(axis='y', labelcolor=ALGO_COLORS['vi'])
        ax.set_title(f'{env.title()} - Model-Based', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Convergence: TD Error (Model-Free) vs Bellman Error (Model-Based)\n(Mean ± IQR across seeds)', 
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'optional' / 'td_error_decay.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"saved: {output_path}")


def plot_state_coverage(results_dir: Path, output_dir: Path):
    """Optional: State space coverage over training
    
    Shows how many unique states have been visited as training progresses.
    VI/PI have full state space coverage from model (shown as horizontal lines).
    1x2 grid: Blackjack | CartPole
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # theoretical state space sizes
    # blackjack: 32 player sums × 11 dealer cards × 2 usable ace = 704 states × 2 actions = 1408
    # but in practice, some states are unreachable (~560 state-action pairs observed)
    # cartpole: discretized, depends on bins (typically ~500-600 reachable)
    
    for col, env in enumerate(['blackjack', 'cartpole']):
        ax = axes[col]
        max_coverage = 0
        
        for algo in ['sarsa', 'qlearning']:
            csv_dir = results_dir / 'raw' / algo / env
            csv_files = list(csv_dir.glob(f'{algo}_{env}_seed*.csv'))
            
            if not csv_files:
                continue
            
            all_coverage = []
            for f in csv_files:
                df = pd.read_csv(f)
                if 'q_table_nonzero' in df.columns:
                    coverage = np.maximum.accumulate(df['q_table_nonzero'].values)
                    all_coverage.append(coverage)
                    max_coverage = max(max_coverage, coverage[-1])
            
            if not all_coverage:
                continue
            
            min_len = min(len(c) for c in all_coverage)
            coverage_array = np.array([c[:min_len] for c in all_coverage])
            
            mean = coverage_array.mean(axis=0)
            q1 = np.percentile(coverage_array, 25, axis=0)
            q3 = np.percentile(coverage_array, 75, axis=0)
            
            episodes = np.arange(len(mean))
            ax.plot(episodes, mean, label=algo.upper(), color=ALGO_COLORS[algo], linewidth=2)
            ax.fill_between(episodes, q1, q3, color=ALGO_COLORS[algo], alpha=0.2)
        
        # VI and PI: full state space coverage from model
        # show as horizontal lines at the max observed coverage
        if max_coverage > 0:
            ax.axhline(y=max_coverage, color=ALGO_COLORS['vi'], linestyle='--', 
                      linewidth=2, alpha=0.8, label=f'VI (full: {int(max_coverage)})')
            ax.axhline(y=max_coverage, color=ALGO_COLORS['pi'], linestyle=':', 
                      linewidth=2, alpha=0.8, label=f'PI (full: {int(max_coverage)})')
        
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Q-Table Entries (state-action pairs)', fontsize=11)
        ax.set_title(f'{env.title()}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('State-Action Space Coverage\n(Mean ± IQR across seeds) · VI/PI have full model access', 
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'optional' / 'state_coverage.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"saved: {output_path}")


def plot_q_value_evolution(results_dir: Path, output_dir: Path):
    """Optional: Q-value statistics evolution over training
    
    Shows mean Q-values as training progresses.
    VI/PI optimal values shown as horizontal reference lines.
    1x2 grid: Blackjack | CartPole
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # VI/PI optimal mean return values (from master_summary.json)
    # these represent the optimal expected value for comparison
    vi_optimal = {'blackjack': -0.038, 'cartpole': None}  # cartpole is steps not return
    pi_optimal = {'blackjack': -0.038, 'cartpole': None}
    
    for col, env in enumerate(['blackjack', 'cartpole']):
        ax = axes[col]
        window = 500 if env == 'blackjack' else 100
        
        for algo in ['sarsa', 'qlearning']:
            csv_dir = results_dir / 'raw' / algo / env
            csv_files = list(csv_dir.glob(f'{algo}_{env}_seed*.csv'))
            
            if not csv_files:
                continue
            
            all_mean_q = []
            for f in csv_files:
                df = pd.read_csv(f)
                if 'mean_q_value' in df.columns:
                    mean_q = df['mean_q_value'].values
                    rolling = pd.Series(mean_q).rolling(window=window, min_periods=1).mean().values
                    all_mean_q.append(rolling)
            
            if not all_mean_q:
                continue
            
            min_len = min(len(q) for q in all_mean_q)
            q_array = np.array([q[:min_len] for q in all_mean_q])
            
            mean = q_array.mean(axis=0)
            q1 = np.percentile(q_array, 25, axis=0)
            q3 = np.percentile(q_array, 75, axis=0)
            
            episodes = np.arange(len(mean))
            ax.plot(episodes, mean, label=algo.upper(), color=ALGO_COLORS[algo], linewidth=2)
            ax.fill_between(episodes, q1, q3, color=ALGO_COLORS[algo], alpha=0.2)
        
        # VI/PI optimal value reference lines (for blackjack only - interpretable)
        if env == 'blackjack' and vi_optimal.get(env) is not None:
            ax.axhline(y=vi_optimal[env], color=ALGO_COLORS['vi'], linestyle='--', 
                      linewidth=2, alpha=0.8, label=f'VI optimal ({vi_optimal[env]:.3f})')
            ax.axhline(y=pi_optimal[env], color=ALGO_COLORS['pi'], linestyle=':', 
                      linewidth=2, alpha=0.8, label=f'PI optimal ({pi_optimal[env]:.3f})')
        
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel(f'Mean Q-Value (w={window})', fontsize=11)
        ax.set_title(f'{env.title()}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right' if env == 'cartpole' else 'upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Q-Value Evolution During Training\n(Mean ± IQR across seeds)', 
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'optional' / 'q_value_evolution.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"saved: {output_path}")


def plot_action_entropy(results_dir: Path, output_dir: Path):
    """Optional: Action entropy (policy confidence) over training
    
    Shows how confident/deterministic the policy becomes.
    High entropy = random actions, Low entropy = confident policy.
    VI/PI are deterministic from the start (entropy ≈ 0).
    1x2 grid: Blackjack | CartPole
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for col, env in enumerate(['blackjack', 'cartpole']):
        ax = axes[col]
        window = 500 if env == 'blackjack' else 100
        
        for algo in ['sarsa', 'qlearning']:
            csv_dir = results_dir / 'raw' / algo / env
            csv_files = list(csv_dir.glob(f'{algo}_{env}_seed*.csv'))
            
            if not csv_files:
                continue
            
            all_entropy = []
            for f in csv_files:
                df = pd.read_csv(f)
                if 'action_entropy' in df.columns:
                    entropy = df['action_entropy'].values
                    rolling = pd.Series(entropy).rolling(window=window, min_periods=1).mean().values
                    all_entropy.append(rolling)
            
            if not all_entropy:
                continue
            
            min_len = min(len(e) for e in all_entropy)
            entropy_array = np.array([e[:min_len] for e in all_entropy])
            
            mean = entropy_array.mean(axis=0)
            q1 = np.percentile(entropy_array, 25, axis=0)
            q3 = np.percentile(entropy_array, 75, axis=0)
            
            episodes = np.arange(len(mean))
            ax.plot(episodes, mean, label=algo.upper(), color=ALGO_COLORS[algo], linewidth=2)
            ax.fill_between(episodes, q1, q3, color=ALGO_COLORS[algo], alpha=0.2)
        
        # VI and PI: deterministic policies (entropy = 0)
        ax.axhline(y=0, color=ALGO_COLORS['vi'], linestyle='--', linewidth=2, 
                   alpha=0.8, label='VI (deterministic)')
        ax.axhline(y=0, color=ALGO_COLORS['pi'], linestyle=':', linewidth=2,
                   alpha=0.8, label='PI (deterministic)')
        
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel(f'Action Entropy (w={window})', fontsize=11)
        ax.set_title(f'{env.title()}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 0.75)  # max entropy for 2 actions is ln(2) ≈ 0.693
    
    plt.suptitle('Policy Confidence (Action Entropy)\n(Mean ± IQR across seeds) · VI/PI are deterministic', 
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'optional' / 'action_entropy.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"saved: {output_path}")


def plot_convergence_all_algorithms(results_dir: Path, output_dir: Path):
    """Convergence curves for ALL algorithms (VI, PI, SARSA, Q-Learning)
    
    2x2 grid showing:
    - Top row: Model-based (VI delta, PI policy changes)
    - Bottom row: Model-free (TD error decay for SARSA vs Q-Learning)
    
    Blackjack and CartPole shown together where applicable.
    """
    fig = plt.figure(figsize=(14, 10))
    
    # === TOP LEFT: VI Convergence (delta decay) ===
    ax1 = fig.add_subplot(2, 2, 1)
    
    for env, style, marker in [('blackjack', '-', 'o'), ('cartpole', '--', 's')]:
        csv_dir = results_dir / 'raw' / 'vi' / env
        csv_files = list(csv_dir.glob('*.csv'))
        if csv_files:
            df = pd.read_csv(csv_files[0])
            ax1.semilogy(df['iteration'], df['delta'], style, color=ALGO_COLORS['vi'], 
                        linewidth=2, marker=marker, markersize=4, markevery=max(1, len(df)//10),
                        label=f'{env.title()}')
    
    ax1.axhline(y=0.0001, color='gray', linestyle=':', alpha=0.7, linewidth=2, label='θ=0.0001')
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Max Value Change (Δ)', fontsize=11)
    ax1.set_title('Value Iteration Convergence', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # === TOP RIGHT: PI Convergence (policy changes) ===
    ax2 = fig.add_subplot(2, 2, 2)
    
    for env, style, marker in [('blackjack', '-', 'o'), ('cartpole', '--', 's')]:
        csv_dir = results_dir / 'raw' / 'pi' / env
        csv_files = list(csv_dir.glob('*.csv'))
        if csv_files:
            df = pd.read_csv(csv_files[0])
            ax2.plot(df['iteration'], df['policy_changes'], style, color=ALGO_COLORS['pi'],
                    linewidth=2, marker=marker, markersize=6,
                    label=f'{env.title()}')
    
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.7, linewidth=2, label='Converged (0 changes)')
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Policy Changes', fontsize=11)
    ax2.set_title('Policy Iteration Convergence', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # === BOTTOM LEFT: Blackjack TD Error (SARSA vs Q-Learning) ===
    ax3 = fig.add_subplot(2, 2, 3)
    
    window = 500
    
    for algo in ['sarsa', 'qlearning']:
        csv_dir = results_dir / 'raw' / algo / 'blackjack'
        csv_files = list(csv_dir.glob(f'{algo}_blackjack_seed*.csv'))
        
        if not csv_files:
            continue
        
        all_td = []
        for f in csv_files:
            df = pd.read_csv(f)
            if 'mean_abs_td_error' in df.columns:
                td_error = df['mean_abs_td_error'].values
                rolling = pd.Series(td_error).rolling(window=window, min_periods=1).mean().values
                all_td.append(rolling)
        
        if not all_td:
            continue
        
        min_len = min(len(t) for t in all_td)
        td_array = np.array([t[:min_len] for t in all_td])
        
        mean = td_array.mean(axis=0)
        q1 = np.percentile(td_array, 25, axis=0)
        q3 = np.percentile(td_array, 75, axis=0)
        
        episodes = np.arange(len(mean))
        ax3.plot(episodes, mean, label=algo.upper(), color=ALGO_COLORS[algo], linewidth=2)
        ax3.fill_between(episodes, q1, q3, color=ALGO_COLORS[algo], alpha=0.2)
    
    ax3.set_xlabel('Episode', fontsize=11)
    ax3.set_ylabel(f'Mean |TD Error| (w={window})', fontsize=11)
    ax3.set_title('Blackjack: Q-Value Convergence', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # === BOTTOM RIGHT: CartPole TD Error (SARSA vs Q-Learning) ===
    ax4 = fig.add_subplot(2, 2, 4)
    
    window = 100
    
    for algo in ['sarsa', 'qlearning']:
        csv_dir = results_dir / 'raw' / algo / 'cartpole'
        csv_files = list(csv_dir.glob(f'{algo}_cartpole_seed*.csv'))
        
        if not csv_files:
            continue
        
        all_td = []
        for f in csv_files:
            df = pd.read_csv(f)
            if 'mean_abs_td_error' in df.columns:
                td_error = df['mean_abs_td_error'].values
                rolling = pd.Series(td_error).rolling(window=window, min_periods=1).mean().values
                all_td.append(rolling)
        
        if not all_td:
            continue
        
        min_len = min(len(t) for t in all_td)
        td_array = np.array([t[:min_len] for t in all_td])
        
        mean = td_array.mean(axis=0)
        q1 = np.percentile(td_array, 25, axis=0)
        q3 = np.percentile(td_array, 75, axis=0)
        
        episodes = np.arange(len(mean))
        ax4.plot(episodes, mean, label=algo.upper(), color=ALGO_COLORS[algo], linewidth=2)
        ax4.fill_between(episodes, q1, q3, color=ALGO_COLORS[algo], alpha=0.2)
    
    ax4.set_xlabel('Episode', fontsize=11)
    ax4.set_ylabel(f'Mean |TD Error| (w={window})', fontsize=11)
    ax4.set_title('CartPole: Q-Value Convergence', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Convergence Rates: All Algorithms\n(Model-based: single seed | Model-free: Mean ± IQR across 40 seeds)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'optional' / 'convergence_all_algorithms.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"saved: {output_path}")


def plot_wall_clock_time(results_dir: Path, output_dir: Path):
    """Optional: Cumulative wall clock time over training
    
    Shows computational cost comparison between all algorithms.
    VI/PI complete in fractions of a second (shown as annotations).
    1x2 grid: Blackjack | CartPole
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for col, env in enumerate(['blackjack', 'cartpole']):
        ax = axes[col]
        
        # model-free algorithms (cumulative over episodes)
        for algo in ['sarsa', 'qlearning']:
            csv_dir = results_dir / 'raw' / algo / env
            csv_files = list(csv_dir.glob(f'{algo}_{env}_seed*.csv'))
            
            if not csv_files:
                continue
            
            all_times = []
            for f in csv_files:
                df = pd.read_csv(f)
                if 'wall_clock_sec' in df.columns:
                    cumtime = np.cumsum(df['wall_clock_sec'].values)
                    all_times.append(cumtime)
            
            if not all_times:
                continue
            
            min_len = min(len(t) for t in all_times)
            time_array = np.array([t[:min_len] for t in all_times])
            
            mean = time_array.mean(axis=0)
            q1 = np.percentile(time_array, 25, axis=0)
            q3 = np.percentile(time_array, 75, axis=0)
            
            episodes = np.arange(len(mean))
            ax.plot(episodes, mean, label=algo.upper(), color=ALGO_COLORS[algo], linewidth=2)
            ax.fill_between(episodes, q1, q3, color=ALGO_COLORS[algo], alpha=0.2)
        
        # VI/PI: get total wall time from CSVs
        vi_time = None
        pi_time = None
        
        for mb_algo, time_var in [('vi', 'vi_time'), ('pi', 'pi_time')]:
            csv_dir = results_dir / 'raw' / mb_algo / env
            csv_files = list(csv_dir.glob('*.csv'))
            if csv_files:
                all_wall = []
                for f in csv_files:
                    df = pd.read_csv(f)
                    if 'wall_time' in df.columns:
                        all_wall.append(df['wall_time'].iloc[-1])
                    elif 'wall_clock_sec' in df.columns:
                        all_wall.append(df['wall_clock_sec'].sum())
                if all_wall:
                    if mb_algo == 'vi':
                        vi_time = np.mean(all_wall)
                    else:
                        pi_time = np.mean(all_wall)
        
        # show VI/PI as horizontal lines (they complete instantly relative to model-free)
        if vi_time is not None:
            ax.axhline(y=vi_time, color=ALGO_COLORS['vi'], linestyle='--', linewidth=2, 
                      alpha=0.8, label=f'VI ({vi_time:.2f}s)')
        if pi_time is not None:
            ax.axhline(y=pi_time, color=ALGO_COLORS['pi'], linestyle=':', linewidth=2,
                      alpha=0.8, label=f'PI ({pi_time:.2f}s)')
        
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Cumulative Time (seconds)', fontsize=11)
        ax.set_title(f'{env.title()}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training Time (Wall Clock)\n(Mean ± IQR across seeds) · VI/PI shown as horizontal lines', 
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'optional' / 'wall_clock_time.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"saved: {output_path}")


def generate_optional_plots(results_dir: Optional[Path] = None, output_dir: Optional[Path] = None):
    """Generate all optional plots (saved to output_dir/optional/)
    
    These plots provide additional insights but are not required for the report.
    Use if you have space or want deeper analysis.
    """
    if results_dir is None:
        results_dir = Path(__file__).parent.parent.parent / 'results'
    if output_dir is None:
        output_dir = results_dir / 'figures'
    
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    
    print("\n" + "="*60)
    print("GENERATING OPTIONAL PLOTS")
    print("="*60)
    
    plot_exploration_decay(results_dir, output_dir)
    plot_td_error_decay(results_dir, output_dir)
    plot_state_coverage(results_dir, output_dir)
    plot_q_value_evolution(results_dir, output_dir)
    plot_action_entropy(results_dir, output_dir)
    plot_convergence_all_algorithms(results_dir, output_dir)
    plot_wall_clock_time(results_dir, output_dir)
    
    print("\n" + "="*60)
    print("OPTIONAL PLOTS COMPLETE")
    print("="*60)
    print(f"location: {output_dir / 'optional'}\n")
    print("OPTIONAL PLOTS GENERATED (7 total):")
    print("  1. exploration_decay.png          - ε decay over episodes")
    print("  2. td_error_decay.png             - TD error (Q-value convergence)")
    print("  3. state_coverage.png             - State-action space exploration")
    print("  4. q_value_evolution.png          - Mean Q-value over training")
    print("  5. action_entropy.png             - Per-episode action diversity")
    print("  6. convergence_all_algorithms.png - Convergence rates (VI/PI/SARSA/QL)")
    print("  7. wall_clock_time.png            - Cumulative training time")


def generate_all_plots(results_dir: Optional[Path] = None, output_dir: Optional[Path] = None):
    """generate all required plots for rl report 
    creates 6 publication-quality visualizations:
    - model-free learning curves (sarsa vs qlearning)
    - blackjack policy heatmaps (all 4 algorithms)
    - cartpole policy heatmaps (angle vs angular velocity)
    - cartpole q-value differences (action preferences)
    - cartpole intuitive summary (key scenarios explained)
    - final performance comparison bar chart
    
    If output.generate_optional_plots is True in config, also generates optional plots.
    """
    if results_dir is None:
        results_dir = Path(__file__).parent.parent.parent / 'results'
    if output_dir is None:
        output_dir = results_dir / 'figures'
    
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # load config to check optional plots flag
    config_path = Path(__file__).parent.parent / 'config' / 'default.yaml'
    generate_optional = False
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        generate_optional = config.get('output', {}).get('generate_optional_plots', False)
    
    print("\n" + "="*60)
    print("GENERATING REPORT PLOTS (8-PAGE OPTIMIZED)")
    print("="*60)
    print(f"results: {results_dir}")
    print(f"output: {output_dir}")
    print(f"optional plots: {'enabled' if generate_optional else 'disabled'}\n")
    
    # NOTE: VI/PI convergence is better as a TABLE in the report text
    # Blackjack: VI=7 iter, PI=4 outer + 26 eval
    # CartPole: VI=918 iter, PI=5 outer + 1846 eval
    # plot_convergence_model_based(results_dir, output_dir)  # DISABLED - use table instead
    
    # === PLOT 1: Model-Free Learning Curves ===
    plot_learning_curves_model_free(results_dir, output_dir)    # 1x2: returns for both envs
    
    # === PLOT 2: Blackjack Policy Heatmaps ===
    plot_blackjack_policy_heatmap(results_dir, output_dir)      # 2x2: all 4 algorithms' policies
    
    # === PLOT 3: CartPole Learned Strategy (SIMPLE & INTUITIVE) ===
    plot_cartpole_simple_visualization(results_dir, output_dir) # 2x2: simple bar charts showing learned strategy
    
    # === PLOT 4: Final Performance Comparison ===
    plot_final_performance_comparison(results_dir, output_dir)  # bar chart: all algos × both envs
    
    # NOTE: CartPole episode length plot REMOVED - redundant with learning curves
    # (reward = steps for CartPole, already shown in plot 1)
    
    print("\n" + "="*60)
    print("COMPLETE - Report Plots Generated")
    print("="*60)
    print(f"location: {output_dir}\n")
    print("PLOTS GENERATED (4 total):")
    print("  1. model_free_learning_curves.png     - SARSA/Q-Learning returns (1x2)")
    print("  2. blackjack_heatmap.png              - Policy heatmap (all 4 algorithms)")
    print("  3. cartpole_learned_strategy.png      - Simple bar charts: learned strategy (2x2)")
    print("  4. final_performance_comparison.png   - Bar chart all algorithms")
    print("\nNOTE: VI/PI convergence data should be presented as a TABLE in your report:")
    print("  | Env       | Algo | Outer Iter | Eval Sweeps | Total |")
    print("  | Blackjack | VI   | 7          | -           | 7     |")
    print("  | Blackjack | PI   | 4          | 26          | 30    |")
    print("  | CartPole  | VI   | 918        | -           | 918   |")
    print("  | CartPole  | PI   | 5          | 1846        | 1851  |")
    
    # === GENERATE CSV SUMMARIES ===
    export_summary_csvs(results_dir, output_dir)
    
    # === GENERATE OPTIONAL PLOTS (if enabled in config) ===
    if generate_optional:
        generate_optional_plots(results_dir, output_dir)


def export_summary_csvs(results_dir: Path, output_dir: Path):
    """Export CSV summary tables for easy report inclusion.
    
    Generates:
    1. model_free_performance.csv - Final performance metrics (SARSA/Q-Learning)
    2. model_based_convergence.csv - VI/PI iteration counts and wall-clock
    3. hyperparameters.csv - All algorithm configurations used
    4. experiment_statistics.csv - Per-algorithm seed counts and wall-clock
    """
    
    csv_dir = output_dir / 'tables'
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n--- Exporting CSV Summary Tables ---")
    
    # === 1. Model-Free Performance Summary ===
    mf_rows = []
    for algo in ['sarsa', 'qlearning']:
        summary_path = results_dir / 'raw' / algo / 'master_summary.json'
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            
            envs = summary.get('environments', {})
            for env in ['blackjack', 'cartpole']:
                env_data = envs.get(env, {})
                if env_data:
                    mean_ret = env_data.get('mean_return', {})
                    wall_time = env_data.get('wall_time_seconds', {})
                    hp = env_data.get('hyperparameters', {})
                    
                    mf_rows.append({
                        'algorithm': algo.upper(),
                        'environment': env.title(),
                        'seeds': env_data.get('num_seeds', 'N/A'),
                        'episodes': hp.get('episodes', 'N/A'),
                        'mean_return': round(mean_ret.get('mean', 0), 4),
                        'std_return': round(mean_ret.get('std', 0), 4),
                        'q1_return': round(mean_ret.get('q1', 0), 4),
                        'median_return': round(mean_ret.get('median', 0), 4),
                        'q3_return': round(mean_ret.get('q3', 0), 4),
                        'iqr_return': round(mean_ret.get('iqr', 0), 4),
                        'mean_wall_clock_sec': round(wall_time.get('mean', 0), 2),
                        'total_wall_clock_sec': round(wall_time.get('mean', 0) * env_data.get('num_seeds', 1), 2),
                    })
    
    if mf_rows:
        mf_df = pd.DataFrame(mf_rows)
        mf_path = csv_dir / 'model_free_performance.csv'
        mf_df.to_csv(mf_path, index=False)
        print(f"  saved: {mf_path}")
    
    # === 2. Model-Based Convergence Summary ===
    mb_rows = []
    for algo in ['vi', 'pi']:
        summary_path = results_dir / 'raw' / algo / 'master_summary.json'
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            
            envs = summary.get('environments', {})
            for env in ['blackjack', 'cartpole']:
                env_data = envs.get(env, {})
                if env_data:
                    # VI/PI store convergence data in 'convergence' sub-dict
                    conv = env_data.get('convergence', {})
                    wall_time = conv.get('wall_time_seconds', {})
                    perf = env_data.get('policy_performance', {})
                    mean_ret = perf.get('mean_return', {})
                    
                    if algo == 'vi':
                        iter_data = conv.get('iterations', {})
                        outer_iter = iter_data.get('mean', 'N/A')
                        eval_sweeps = '-'
                        total_work = outer_iter
                        delta_data = conv.get('final_delta', {})
                    else:  # pi
                        pol_iter = conv.get('iterations', {})  # PI uses 'iterations' for policy iterations
                        eval_iter = conv.get('total_eval_iterations', {})  # PI uses 'total_eval_iterations'
                        outer_iter = pol_iter.get('mean', 'N/A')
                        eval_sweeps = eval_iter.get('mean', 'N/A')
                        if isinstance(outer_iter, (int, float)) and isinstance(eval_sweeps, (int, float)):
                            total_work = int(outer_iter) + int(eval_sweeps)
                        else:
                            total_work = 'N/A'
                        delta_data = conv.get('final_delta', {})
                    
                    mb_rows.append({
                        'algorithm': algo.upper(),
                        'environment': env.title(),
                        'outer_iterations': int(outer_iter) if isinstance(outer_iter, (int, float)) else outer_iter,
                        'eval_sweeps': int(eval_sweeps) if isinstance(eval_sweeps, (int, float)) else eval_sweeps,
                        'total_work': int(total_work) if isinstance(total_work, (int, float)) else total_work,
                        'mean_wall_clock_sec': round(wall_time.get('mean', 0), 4),
                        'final_delta': f"{delta_data.get('mean', 0):.2e}" if isinstance(delta_data, dict) else 'N/A',
                        'mean_return': round(mean_ret.get('mean', 0), 4),
                        'seeds': env_data.get('num_seeds', 'N/A'),
                    })
    
    if mb_rows:
        mb_df = pd.DataFrame(mb_rows)
        mb_path = csv_dir / 'model_based_convergence.csv'
        mb_df.to_csv(mb_path, index=False)
        print(f"  saved: {mb_path}")
    
    # === 3. Hyperparameters Summary ===
    config_path = Path(__file__).parent.parent / 'config' / 'default.yaml'
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        hp_rows = []
        
        # shared hyperparameters
        shared = config.get('hyperparameters', {})
        hp_rows.append({'scope': 'shared', 'parameter': 'gamma', 'value': shared.get('gamma', 'N/A'), 'description': 'discount factor'})
        hp_rows.append({'scope': 'shared', 'parameter': 'episodes', 'value': shared.get('episodes', 'N/A'), 'description': 'default training episodes'})
        
        # algorithm-specific
        for algo in ['sarsa', 'qlearning', 'vi', 'pi']:
            algo_config = config.get(algo, {})
            for param, value in algo_config.items():
                hp_rows.append({'scope': algo.upper(), 'parameter': param, 'value': value, 'description': ''})
        
        # environment-specific
        for env in ['blackjack', 'cartpole']:
            env_config = config.get(env, {})
            for param, value in env_config.items():
                hp_rows.append({'scope': env.title(), 'parameter': param, 'value': str(value), 'description': ''})
        
        hp_df = pd.DataFrame(hp_rows)
        hp_path = csv_dir / 'hyperparameters.csv'
        hp_df.to_csv(hp_path, index=False)
        print(f"  saved: {hp_path}")
    
    # === 4. Experiment Statistics (all algorithms) ===
    stats_rows = []
    for algo in ['sarsa', 'qlearning', 'vi', 'pi']:
        summary_path = results_dir / 'raw' / algo / 'master_summary.json'
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            
            envs = summary.get('environments', {})
            for env in ['blackjack', 'cartpole']:
                env_data = envs.get(env, {})
                if env_data:
                    wall_time = env_data.get('wall_time_seconds', env_data.get('convergence', {}).get('wall_time_seconds', {}))
                    hp = env_data.get('hyperparameters', {})
                    num_seeds = env_data.get('num_seeds', 0)
                    episodes = hp.get('episodes', 1) if algo in ['sarsa', 'qlearning'] else '-'
                    timestamps = env_data.get('run_timestamps', {})
                    
                    stats_rows.append({
                        'algorithm': algo.upper(),
                        'environment': env.title(),
                        'seeds_completed': num_seeds,
                        'episodes_per_seed': episodes,
                        'total_episodes': num_seeds * episodes if isinstance(episodes, int) else '-',
                        'mean_sec_per_seed': round(wall_time.get('mean', 0), 2),
                        'total_wall_clock_sec': round(wall_time.get('mean', 0) * num_seeds, 2),
                        'first_run_utc': timestamps.get('first', 'N/A'),
                        'last_run_utc': timestamps.get('last', 'N/A'),
                    })
    
    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        stats_path = csv_dir / 'experiment_statistics.csv'
        stats_df.to_csv(stats_path, index=False)
        print(f"  saved: {stats_path}")
    
    print(f"\nCSV tables saved to: {csv_dir}")


def plot_cartpole_simple_visualization(results_dir: Path, output_dir: Path):
    """CartPole policy heatmap visualization matching blackjack style.
    
    Creates a 2x2 grid showing each algorithm's learned policy:
    - X-axis: Pole angle θ (degrees)
    - Y-axis: Angular velocity θ̇ (degrees/second)
    - Color: Blue = Push Left, Red = Push Right
    - Text: L = Left, R = Right (majority action)
    - Gray background: Unvisited states (model-free only)
    """
    
    # create environment
    env = gym.make('CartPole-v1')
    
    # algorithm configs - same order as blackjack (VI, PI top row; SARSA, Q-Learning bottom)
    # bins differ: VI/PI use [4,4,10,12], SARSA/Q-Learning use [3,3,8,12]
    algos = [
        ('vi', 'Value Iteration (VI)', True, [4, 4, 10, 12]),
        ('pi', 'Policy Iteration (PI)', True, [4, 4, 10, 12]),
        ('sarsa', 'SARSA (On-Policy)', False, [3, 3, 8, 12]),
        ('qlearning', 'Q-Learning (Off-Policy)', False, [3, 3, 8, 12]),
    ]
    
    # custom diverging colormap: blue (Push Left) - white - red (Push Right)
    # using a lighter version for better readability
    cmap = plt.cm.coolwarm  # lighter blue-white-red colormap
    
    # figure sized to accommodate heatmaps with minimal whitespace
    fig, axes = plt.subplots(2, 2, figsize=(14, 16))
    axes = axes.flatten()
    plt.subplots_adjust(wspace=0.25, hspace=0.20, left=0.08, right=0.88)
    
    for idx, (algo, algo_name, is_model_based, bins) in enumerate(algos):
        ax = axes[idx]
        
        # create algorithm-specific discretizer
        discretizer = StateDiscretizer(env, bins=bins)
        n_theta = discretizer.bins[2]      # angle bins (8 or 10)
        n_thetadot = discretizer.bins[3]   # angular velocity bins (12)
        
        # get theta and theta_dot bin edges for axis labels
        theta_edges = discretizer.bin_edges[2]
        thetadot_edges = discretizer.bin_edges[3]
        
        # load policy (data is in results/raw/{algo}/cartpole/)
        data_dir = results_dir / 'raw' / algo / 'cartpole'
        policy_files = sorted(data_dir.glob(f'{algo}_cartpole_seed*_policy.npy'))
        
        if not policy_files:
            ax.text(0.5, 0.5, f'No data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            continue
        
        policy_data = np.load(policy_files[0], allow_pickle=True)
        
        # build policy matrix: theta (angle) x theta_dot (angular velocity)
        # aggregate over cart position (x) and cart velocity (x_dot)
        policy_matrix = np.full((n_thetadot, n_theta), np.nan)
        count_matrix = np.zeros((n_thetadot, n_theta))
        
        if is_model_based:
            # model-based: all states have a policy
            for x_idx in range(discretizer.bins[0]):
                for xdot_idx in range(discretizer.bins[1]):
                    for theta_idx in range(n_theta):
                        for thetadot_idx in range(n_thetadot):
                            state_idx = discretizer.tuple_to_index((x_idx, xdot_idx, theta_idx, thetadot_idx))
                            action = int(policy_data[state_idx])
                            if np.isnan(policy_matrix[thetadot_idx, theta_idx]):
                                policy_matrix[thetadot_idx, theta_idx] = 0
                            policy_matrix[thetadot_idx, theta_idx] += action
                            count_matrix[thetadot_idx, theta_idx] += 1
            policy_matrix = policy_matrix / count_matrix
        else:
            # model-free: only visited states
            policy_dict = policy_data.item() if hasattr(policy_data, 'item') else policy_data
            for state_tuple, action in policy_dict.items():
                x_idx, xdot_idx, theta_idx, thetadot_idx = state_tuple
                if np.isnan(policy_matrix[thetadot_idx, theta_idx]):
                    policy_matrix[thetadot_idx, theta_idx] = 0
                    count_matrix[thetadot_idx, theta_idx] = 0
                policy_matrix[thetadot_idx, theta_idx] += action
                count_matrix[thetadot_idx, theta_idx] += 1
            mask = count_matrix > 0
            policy_matrix[mask] = policy_matrix[mask] / count_matrix[mask]
        
        # create heatmap with proper masked array
        masked_policy = np.ma.masked_invalid(policy_matrix)
        
        # set gray background for unvisited states
        ax.set_facecolor('#e0e0e0')
        
        im = ax.imshow(masked_policy, cmap=cmap, vmin=0, vmax=1, 
                       aspect='auto', origin='lower', interpolation='nearest')
        
        # add text labels showing L (left) or R (right) for each cell
        for i in range(n_thetadot):
            for j in range(n_theta):
                if not np.isnan(policy_matrix[i, j]):
                    action_val = policy_matrix[i, j]
                    # show L or R based on majority action
                    label = 'R' if action_val > 0.5 else 'L'
                    # text color: black for readability
                    ax.text(j, i, label, ha='center', va='center',
                           fontsize=9, fontweight='bold', color='black')
        
        # x-axis: theta (pole angle)
        ax.set_xticks(range(n_theta))
        theta_labels = [f'{np.degrees((theta_edges[i]+theta_edges[i+1])/2):.0f}°' 
                       for i in range(n_theta)]
        ax.set_xticklabels(theta_labels, fontsize=11)
        ax.set_xlabel('Pole Angle θ', fontsize=13, fontweight='bold')
        
        # y-axis: theta_dot (angular velocity)  
        ax.set_yticks(range(n_thetadot))
        thetadot_labels = [f'{np.degrees((thetadot_edges[i]+thetadot_edges[i+1])/2):.0f}°/s' 
                         for i in range(n_thetadot)]
        ax.set_yticklabels(thetadot_labels, fontsize=10)
        ax.set_ylabel('Angular Velocity θ̇', fontsize=13, fontweight='bold')
        
        ax.set_title(algo_name, fontsize=15, fontweight='bold', pad=10)
        
        # add grid lines for clarity
        for x in np.arange(-0.5, n_theta, 1):
            ax.axvline(x, color='gray', linewidth=0.5, alpha=0.5)
        for y in np.arange(-0.5, n_thetadot, 1):
            ax.axhline(y, color='gray', linewidth=0.5, alpha=0.5)
    
    # vertical colorbar on the right
    cbar_ax = fig.add_axes([0.90, 0.08, 0.02, 0.84])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Push Left (L)', 'Mixed', 'Push Right (R)'])
    cbar.ax.tick_params(labelsize=11, labelrotation=90)
    cbar.set_label('Action Probability (Aggregated over cart position/velocity)', fontsize=12)
    
    # main title and subtitle
    fig.text(0.48, 0.97, 'CartPole Learned Policy Heatmaps',
            ha='center', fontsize=20, fontweight='bold')
    fig.text(0.48, 0.955, 'L = Push Left  |  R = Push Right  |  Color = Action Probability  |  Gray = Unvisited States',
            ha='center', fontsize=12, style='italic')
    
    plt.subplots_adjust(top=0.92)
    output_path = output_dir / 'cartpole_learned_strategy.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"saved: {output_path}")


if __name__ == '__main__':
    generate_all_plots()
