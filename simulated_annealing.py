import numpy as np
import random
import copy

from case_data import *
from timer import Timer


class SimulatedAnnealing:
    """
    Simulated Annealing algorithm for Job Shop Scheduling
    Can be used to improve existing solutions
    """
    
    def __init__(self, case_id, initial_order=None, verbose=False):
        self.case_id = case_id
        self.verbose = verbose
        
        # Case parameters
        self.batch_size = MIN_BATCH[case_id]
        self.Nm = MACHINES[1:]
        self.Nj = list(map(sum, JOBS[case_id]))
        self.jgsj = PROCESS_TIME[case_id]
        self.hmsj = CHANGE_TIME[case_id].transpose(2, 0, 1)
        self.max_batch_num = MAX_BATCH_NUM[case_id]
        
        # SA parameters
        self.initial_temp = 10000
        self.final_temp = 1
        self.cooling_rate = 0.95
        self.iterations_per_temp = 50
        
        self.current_order = initial_order
        self.best_order = None
        self.best_time = float('inf')
        
    def evaluate_order(self, order):
        """Calculate makespan for a given order"""
        # Calculate time efficiently
        t = np.zeros((len(self.Nj), self.max_batch_num, len(self.Nm) + 1, max(self.Nm)))
        
        for s in range(1, len(self.Nm) + 1):
            for m in range(self.Nm[s - 1]):
                time = 0
                prev_job = None
                stage_order = order[s - 1][m]
                
                for j, b in stage_order:
                    p_time = JOBS[self.case_id][j][b] * self.jgsj[j, s - 1]
                    
                    if p_time != 0:
                        if prev_job is not None and j != prev_job:
                            time += self.hmsj[s - 1, prev_job, j] * 1000
                        
                        # Get previous stage time
                        prev_stage_time = 0
                        if s > 1:
                            for prev_s in range(s - 1, 0, -1):
                                for prev_m in range(self.Nm[prev_s - 1]):
                                    if t[j, b, prev_s, prev_m] > 0:
                                        prev_stage_time = t[j, b, prev_s, prev_m]
                                        break
                                if prev_stage_time > 0:
                                    break
                        
                        time = max(time, prev_stage_time)
                        time += p_time
                        t[j, b, s, m] = time
                        prev_job = j
                    else:
                        # Get time from previous stage
                        prev_stage_time = 0
                        if s > 1:
                            for prev_s in range(s - 1, 0, -1):
                                for prev_m in range(self.Nm[prev_s - 1]):
                                    if t[j, b, prev_s, prev_m] > 0:
                                        prev_stage_time = t[j, b, prev_s, prev_m]
                                        break
                                if prev_stage_time > 0:
                                    break
                        t[j, b, s, m] = prev_stage_time
        
        return np.max(t)
    
    def generate_neighbor(self, order):
        """Generate a neighbor solution by swapping two jobs in a random machine"""
        new_order = copy.deepcopy(order)
        
        # Choose a random stage and machine
        stage = random.randint(0, len(self.Nm) - 1)
        machine = random.randint(0, self.Nm[stage] - 1)
        
        # Get the order for this machine
        machine_order = new_order[stage][machine]
        
        if len(machine_order) >= 2:
            # Swap two random positions
            pos1, pos2 = random.sample(range(len(machine_order)), 2)
            machine_order[pos1], machine_order[pos2] = machine_order[pos2], machine_order[pos1]
        
        return new_order
    
    def run(self, max_iterations=1000):
        """Run simulated annealing"""
        if self.current_order is None:
            raise ValueError("Initial order must be provided")
        
        current_time = self.evaluate_order(self.current_order)
        self.best_order = copy.deepcopy(self.current_order)
        self.best_time = current_time
        
        temp = self.initial_temp
        iteration = 0
        
        while temp > self.final_temp and iteration < max_iterations:
            for _ in range(self.iterations_per_temp):
                # Generate neighbor
                neighbor_order = self.generate_neighbor(self.current_order)
                neighbor_time = self.evaluate_order(neighbor_order)
                
                # Calculate acceptance probability
                delta = neighbor_time - current_time
                if delta < 0:
                    # Better solution, always accept
                    self.current_order = neighbor_order
                    current_time = neighbor_time
                    
                    if current_time < self.best_time:
                        self.best_order = copy.deepcopy(self.current_order)
                        self.best_time = current_time
                        if self.verbose:
                            print(f"New best: {self.best_time}")
                else:
                    # Worse solution, accept with probability
                    acceptance_prob = np.exp(-delta / temp)
                    if random.random() < acceptance_prob:
                        self.current_order = neighbor_order
                        current_time = neighbor_time
                
                iteration += 1
            
            # Cool down
            temp *= self.cooling_rate
            
            if iteration % 100 == 0 and self.verbose:
                print(f"Iteration {iteration}, temp {temp:.2f}, best {self.best_time}")
        
        return self.best_order
