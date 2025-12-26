"""
Test script for hybrid optimization approach
"""

from orders import ORDERS
from timer import Timer
from ants_algo import AntsAlgorithm
from hybrid_optimizer import HybridOptimizer
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


ID = 4

print("="*60)
print(f"Job Shop Scheduling Optimization - Case {ID}")
print("="*60)

# Method 1: Original ACO
print("\n[Method 1] Original Ant Colony Optimization (500 epochs)")
print("-" * 60)
aa_original = AntsAlgorithm(case_id=ID, verbose=False, random_mode=False)
# Restore original parameters for comparison
aa_original.ant_per_epoch = 200
aa_original.elite_ants = 0  # Disable elite ants for original
aa_order_original = aa_original.run(epoch_num=500)
print(f"Best time: {aa_original.bstime}")

# Method 2: Improved ACO with elite ants and heuristics
print("\n[Method 2] Improved Ant Colony Optimization")
print("-" * 60)
aa_improved = AntsAlgorithm(case_id=ID, verbose=False, random_mode=False)
aa_order_improved = aa_improved.run(epoch_num=500)
print(f"Best time: {aa_improved.bstime}")

# Method 3: Hybrid approach
print("\n[Method 3] Hybrid Optimizer (Greedy + Improved ACO)")
print("-" * 60)
hybrid = HybridOptimizer(case_id=ID, verbose=True)
hybrid_order, hybrid_time = hybrid.run(aco_epochs=300, use_greedy_init=True)

# Summary comparison
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(f"Original ACO (500 epochs):     {aa_original.bstime:10.0f}")
print(f"Improved ACO (500 epochs):     {aa_improved.bstime:10.0f}")
print(f"Hybrid Optimizer:              {hybrid_time:10.0f}")
print("="*60)

improvement_over_original = ((aa_original.bstime - hybrid_time) / aa_original.bstime * 100)
print(f"\nImprovement over original: {improvement_over_original:.2f}%")

# Use the best solution for further testing
best_order = None
best_time = float('inf')
best_method = ""

if aa_original.bstime < best_time:
    best_time = aa_original.bstime
    best_order = aa_order_original
    best_method = "Original ACO"

if aa_improved.bstime < best_time:
    best_time = aa_improved.bstime
    best_order = aa_order_improved
    best_method = "Improved ACO"

if hybrid_time < best_time:
    best_time = hybrid_time
    best_order = hybrid_order
    best_method = "Hybrid Optimizer"

print(f"\nBest method: {best_method} with time {best_time}")

# Verify the best solution
print(f"\nVerifying best solution...")
timer = Timer(case_id=ID, orders=best_order, random_mode=False)
timer.get_total_time(repeat=1)
