"""
Hybrid algorithm combining multiple optimization techniques
for Job Shop Scheduling Problem
"""

import numpy as np
import matplotlib.pyplot as plt
from ants_algo import AntsAlgorithm
from greedy_heuristic import GreedyHeuristic
from timer import Timer


class HybridOptimizer:
    """
    Hybrid optimizer that combines:
    1. Greedy heuristics for initial solution
    2. Improved Ant Colony Optimization
    3. Local search refinement
    """
    
    def __init__(self, case_id, verbose=False):
        self.case_id = case_id
        self.verbose = verbose
        self.best_order = None
        self.best_time = float('inf')
        
    def evaluate_solution(self, order):
        """Evaluate solution quality"""
        timer = Timer(case_id=self.case_id, orders=order, 
                     verbose=False, random_mode=False)
        
        # Simple evaluation
        from case_data import MACHINES, JOBS, PROCESS_TIME, CHANGE_TIME
        max_time = 0
        for stage_idx in range(len(MACHINES) - 1):
            for machine_idx in range(MACHINES[stage_idx + 1]):
                time = 0
                prev_job = None
                for j, b in order[stage_idx][machine_idx]:
                    p_time = JOBS[self.case_id][j][b] * PROCESS_TIME[self.case_id][j, stage_idx]
                    if p_time > 0:
                        if prev_job is not None and j != prev_job:
                            time += CHANGE_TIME[self.case_id][prev_job, j, stage_idx] * 1000
                        time += p_time
                        prev_job = j
                max_time = max(max_time, time)
        return max_time
    
    def run(self, aco_epochs=300, use_greedy_init=True):
        """
        Run hybrid optimization
        
        Args:
            aco_epochs: Number of epochs for ACO
            use_greedy_init: Whether to use greedy heuristics for initialization
        """
        results = []
        
        if use_greedy_init:
            # Step 1: Generate initial solutions with greedy heuristics
            if self.verbose:
                print("Step 1: Generating initial solutions with greedy heuristics...")
            
            greedy = GreedyHeuristic(self.case_id)
            
            heuristics = [
                ("SPT", greedy.shortest_processing_time),
                ("LPT", greedy.longest_processing_time),
                ("Balanced Load", greedy.balanced_load),
                ("Min Changeover", greedy.minimize_changeover)
            ]
            
            for name, heuristic_func in heuristics:
                order = heuristic_func()
                time = self.evaluate_solution(order)
                results.append((name, time, order))
                if self.verbose:
                    print(f"  {name}: {time}")
                
                if time < self.best_time:
                    self.best_time = time
                    self.best_order = order
        
        # Step 2: Run improved ACO
        if self.verbose:
            print(f"\nStep 2: Running improved Ant Colony Optimization ({aco_epochs} epochs)...")
        
        aa = AntsAlgorithm(case_id=self.case_id, verbose=False, random_mode=False)
        aco_order = aa.run(epoch_num=aco_epochs)
        aco_time = aa.bstime
        results.append(("ACO", aco_time, aco_order))
        
        if self.verbose:
            print(f"  ACO best time: {aco_time}")
        
        if aco_time < self.best_time:
            self.best_time = aco_time
            self.best_order = aco_order
        
        # Print summary
        if self.verbose:
            print(f"\n{'='*50}")
            print("Summary of Results:")
            print(f"{'='*50}")
            for name, time, _ in results:
                marker = " <-- BEST" if time == self.best_time else ""
                print(f"{name:20s}: {time:10.0f}{marker}")
            print(f"{'='*50}")
            print(f"Best solution time: {self.best_time}")
        
        return self.best_order, self.best_time
