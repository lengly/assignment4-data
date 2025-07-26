#!/usr/bin/env python3
"""
Chinchilla Scaling Law Experiment Planning
Based on IsoFLOP Profile method

Total compute budget: 3e19 FLOPs
Experiment goal: Explore optimal (N, D) combinations and establish scaling law
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

class ChinchillaScalingExperiment:
    def __init__(self, total_flops=3e19):
        self.total_flops = total_flops
        self.experiments = []
        
    def compute_tokens_from_params_flops(self, N, C):
        """Calculate token count from parameter count and FLOPs: C ≈ 6ND"""
        return int(C / (6 * N))
    
    def compute_params_from_tokens_flops(self, D, C):
        """Calculate parameter count from token count and FLOPs: C ≈ 6ND"""
        return int(C / (6 * D))
    
    def design_isoflop_experiments(self):
        """Design IsoFLOP profile experiments
        
        Strategy:
        1. Choose several different FLOPs budget points
        2. For each FLOPs budget, explore different (N, D) combinations
        3. Cover the range from small to large models
        """
        
        # Step 1: Choose different FLOPs budget points
        # Start from a small budget and gradually increase to the target budget
        flops_budgets = [
            1e17,   # 0.1e18
            3e17,   # 0.3e18  
            1e18,   # 1e18
            3e18,   # 3e18
            1e19,   # 10e18
            #3e19,   # 30e18 (target budget)
        ]
        
        experiments = []

        for flops_budget in flops_budgets:
            # For each FLOPs budget, design multiple (N, D) combinations
            # Parameter count range: from small to large models
            param_counts = self._get_param_range_for_flops(flops_budget)
            
            for N in param_counts:
                D = self.compute_tokens_from_params_flops(N, flops_budget)
                
                # Calculate batch size and iteration count
                batch_size, iterations = self._compute_batch_size_and_iterations(N, D)
                
                experiments.append({
                    'experiment_id': len(experiments) + 1,
                    'flops_budget': flops_budget,
                    'model_size_params': N,
                    'training_tokens': D,
                    'tokens_per_param': int(D / N),
                    'batch_size': batch_size,
                    'iterations': iterations,
                    'estimated_training_time_hours': self._estimate_training_time(N, D)
                })
        
        self.experiments = experiments
        total_hours = sum(e['estimated_training_time_hours'] for e in experiments)
        total_flops = sum(e['flops_budget'] for e in experiments)
        print(f"Total hours: {total_hours}")
        print(f"Total flops: {total_flops}")
        for e in experiments:
            print(e)
        return experiments
    
    def _get_param_range_for_flops(self, flops_budget):
        """Generate a reasonable parameter count range for a given FLOPs budget"""
        chinchilla_ratio = 20
        ratio = np.array([5, 10, 20, 40, 60]) if flops_budget < 1e19 else np.array([20])
        param_counts = (flops_budget / (6 * ratio)) ** 0.5
        return param_counts.astype(int)
    
    def _compute_batch_size_and_iterations(self, N, D):
        """Calculate batch size and iteration count
        
        Rule:
        - If parameter count <= 120M, batch_size = 64*1024*4 = 262144
        - If parameter count > 120M, batch_size = 64*1024*4*2 = 524288
        """
        if N <= 120e6:  # 120M parameters
            batch_size = 64 * 1024 * 4  # 262144
        else:
            batch_size = 64 * 1024 * 4 * 2  # 524288
        
        iterations = int(D / batch_size)
        
        return batch_size, iterations
    
    def _estimate_training_time(self, N, D):
        """Estimate training time (hours)
        
        Assumptions:
        - Using 1 H100 GPU
        - MFU (Model FLOPs Utilization) = 20%
        - H100 peak performance ~ 989 TFLOPS (mixed precision)
        """
        # Calculate total FLOPs
        total_flops = 6 * float(N) * float(D)
        
        # Hardware config
        num_gpus = 1
        h100_peak_flops = 989e12  # TFLOPS
        mfu = 0.2
        
        # Effective compute
        effective_flops_per_second = num_gpus * h100_peak_flops * mfu
        
        # Training time (seconds)
        training_time_seconds = total_flops / effective_flops_per_second
        
        # Convert to hours
        training_time_hours = training_time_seconds / 3600
        
        return round(training_time_hours, 1)
    
    def print_experiment_plan(self):
        """Print experiment plan"""
        if not self.experiments:
            self.design_isoflop_experiments()
        if not self.experiments:
            print("[Error] No experiments generated, please check the design logic!")
            return
        
        df = pd.DataFrame(self.experiments)
        print(f"[Debug] DataFrame columns: {df.columns.tolist()}")
        
        print("=" * 80)
        print("Chinchilla Scaling Law Experiment Plan")
        print(f"Total compute budget: {self.total_flops:.2e} FLOPs")
        print(f"Number of planned experiments: {len(self.experiments)}")
        print("=" * 80)
        
        # Display by FLOPs budget group
        df = pd.DataFrame(self.experiments)
        
        for flops_budget in df['flops_budget'].unique():
            budget_experiments = df[df['flops_budget'] == flops_budget]
            
            print(f"\n📊 FLOPs budget: {flops_budget:.2e}")
            print(f"   Number of experiments: {len(budget_experiments)}")
            print("-" * 60)
            
            for _, exp in budget_experiments.iterrows():
                # Convert parameter and token count to M units
                params_M = int(exp['model_size_params'] / 1e6)
                tokens_M = int(exp['training_tokens'] / 1e6)
                batch_size_K = int(exp['batch_size'] / 1e3)
                
                print(f"Experiment {int(exp['experiment_id']):2d}: "
                      f"N={params_M:4d}M params, "
                      f"D={tokens_M:4d}M tokens, "
                      f"Ratio={exp['tokens_per_param']:5.1f} tokens/param, "
                      f"batch_size={batch_size_K}K, "
                      f"iterations={exp['iterations']}, "
                      f"Estimated {exp['estimated_training_time_hours']:4.1f} hours")
        
        # Show summary info
        total_hours = df['estimated_training_time_hours'].sum()
        params_min_M = int(df['model_size_params'].min() / 1e6)
        params_max_M = int(df['model_size_params'].max() / 1e6)
        tokens_min_M = int(df['training_tokens'].min() / 1e6)
        tokens_max_M = int(df['training_tokens'].max() / 1e6)
        
        print(f"\n📈 Experiment summary:")
        print(f"   - Total number of experiments: {len(self.experiments)}")
        print(f"   - Parameter count range: {params_min_M}M ~ {params_max_M}M")
        print(f"   - Token count range: {tokens_min_M}M ~ {tokens_max_M}M")
        print(f"   - Tokens/Param ratio range: {df['tokens_per_param'].min():.1f} ~ {df['tokens_per_param'].max():.1f}")
        print(f"   - Estimated total training time: {total_hours:.1f} hours (assuming 1×H100 MFU=20%)")
        
        # Parallel training suggestions
        print(f"\n🚀 Parallel training suggestions:")
        print(f"   - If you have multiple GPU clusters, you can run multiple experiments simultaneously")
        print(f"   - It is recommended to run small FLOPs budget experiments first to quickly obtain preliminary results")
        print(f"   - Large model experiments can be started after analyzing small model results")
        
    def save_experiment_plan(self, filename="pretrain/scaling_law_experiments.csv"):
        """Save experiment plan to CSV file"""
        if not self.experiments:
            self.design_isoflop_experiments()
        
        df = pd.DataFrame(self.experiments)
        df.to_csv(filename, index=False)
        print(f"\n💾 Experiment plan saved to: {filename}")
        
    def plot_experiment_design(self):
        """Visualize experiment design"""
        if not self.experiments:
            self.design_isoflop_experiments()
        
        df = pd.DataFrame(self.experiments)
        
        plt.figure(figsize=(12, 8))
        
        # Use different colors for different FLOPs budgets
        colors = plt.cm.viridis(np.linspace(0, 1, len(df['flops_budget'].unique())))
        
        for i, flops_budget in enumerate(df['flops_budget'].unique()):
            budget_data = df[df['flops_budget'] == flops_budget]
            plt.scatter(budget_data['model_size_params'], 
                       budget_data['training_tokens'],
                       c=[colors[i]], 
                       label=f'FLOPs={flops_budget:.1e}',
                       s=60, alpha=0.7)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Parameter Count (N)')
        plt.ylabel('Training Tokens (D)')
        plt.title('Chinchilla Scaling Law Experiment Design\nIsoFLOP Profile')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add theoretical optimal line (20 tokens per parameter)
        param_range = np.logspace(6, 12, 100)
        optimal_tokens = param_range * 20
        plt.plot(param_range, optimal_tokens, 'r--', alpha=0.5, 
                linewidth=2, label='Theoretical Optimal (20 tokens/param)')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig("pretrain/scaling_law_experiments.png")
        print("[Image saved to pretrain/scaling_law_experiments.png]")
        plt.show()

def main():
    # Create experiment planner
    experiment = ChinchillaScalingExperiment(total_flops=3e19)
    
    # Design experiments
    experiment.design_isoflop_experiments()
    
    # Print experiment plan
    experiment.print_experiment_plan()
    
    # Save experiment plan
    experiment.save_experiment_plan()
    
    # Visualize experiment design
    experiment.plot_experiment_design()

if __name__ == "__main__":
    main()
