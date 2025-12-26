import numpy as np
from case_data import *


class GreedyHeuristic:
    """
    Greedy heuristic algorithms for Job Shop Scheduling
    Implements various priority rules for quick initial solutions
    """
    
    def __init__(self, case_id):
        self.case_id = case_id
        self.batch_size = MIN_BATCH[case_id]
        self.Nm = MACHINES[1:]
        self.Nj = list(map(sum, JOBS[case_id]))
        self.jgsj = PROCESS_TIME[case_id]
        self.hmsj = CHANGE_TIME[case_id].transpose(2, 0, 1)
        self.jobs = JOBS[case_id]
        
    def shortest_processing_time(self):
        """SPT: Schedule jobs with shortest processing time first"""
        order = [[] for _ in range(len(self.Nm))]
        for stage_idx in range(len(self.Nm)):
            for machine_idx in range(self.Nm[stage_idx]):
                order[stage_idx].append([])
        
        # Create job list with processing times
        job_list = []
        for j in range(len(self.Nj)):
            for b in range(len(self.jobs[j])):
                for stage_idx in range(len(self.Nm)):
                    p_time = self.jobs[j][b] * self.jgsj[j, stage_idx]
                    if p_time > 0:
                        job_list.append((j, b, stage_idx, p_time))
        
        # Sort by processing time
        job_list.sort(key=lambda x: x[3])
        
        # Assign jobs to machines in round-robin fashion
        machine_counters = [[0 for _ in range(self.Nm[s])] for s in range(len(self.Nm))]
        
        for j, b, stage_idx, p_time in job_list:
            # Find machine with minimum load
            min_machine = 0
            min_load = machine_counters[stage_idx][0]
            for m in range(self.Nm[stage_idx]):
                if machine_counters[stage_idx][m] < min_load:
                    min_load = machine_counters[stage_idx][m]
                    min_machine = m
            
            order[stage_idx][min_machine].append((j, b))
            machine_counters[stage_idx][min_machine] += p_time
        
        return order
    
    def longest_processing_time(self):
        """LPT: Schedule jobs with longest processing time first"""
        order = [[] for _ in range(len(self.Nm))]
        for stage_idx in range(len(self.Nm)):
            for machine_idx in range(self.Nm[stage_idx]):
                order[stage_idx].append([])
        
        # Create job list with processing times
        job_list = []
        for j in range(len(self.Nj)):
            for b in range(len(self.jobs[j])):
                for stage_idx in range(len(self.Nm)):
                    p_time = self.jobs[j][b] * self.jgsj[j, stage_idx]
                    if p_time > 0:
                        job_list.append((j, b, stage_idx, p_time))
        
        # Sort by processing time (descending)
        job_list.sort(key=lambda x: x[3], reverse=True)
        
        # Assign jobs to machines with minimum load
        machine_counters = [[0 for _ in range(self.Nm[s])] for s in range(len(self.Nm))]
        
        for j, b, stage_idx, p_time in job_list:
            # Find machine with minimum load
            min_machine = 0
            min_load = machine_counters[stage_idx][0]
            for m in range(self.Nm[stage_idx]):
                if machine_counters[stage_idx][m] < min_load:
                    min_load = machine_counters[stage_idx][m]
                    min_machine = m
            
            order[stage_idx][min_machine].append((j, b))
            machine_counters[stage_idx][min_machine] += p_time
        
        return order
    
    def minimize_changeover(self):
        """Group similar jobs together to minimize changeover times"""
        order = [[] for _ in range(len(self.Nm))]
        for stage_idx in range(len(self.Nm)):
            for machine_idx in range(self.Nm[stage_idx]):
                order[stage_idx].append([])
        
        # For each stage and machine, group jobs by type
        for stage_idx in range(len(self.Nm)):
            # Collect all jobs for this stage
            stage_jobs = []
            for j in range(len(self.Nj)):
                for b in range(len(self.jobs[j])):
                    p_time = self.jobs[j][b] * self.jgsj[j, stage_idx]
                    if p_time > 0:
                        stage_jobs.append((j, b, p_time))
            
            # Sort by job type to group similar jobs
            stage_jobs.sort(key=lambda x: (x[0], x[1]))
            
            # Distribute to machines in round-robin
            machine_idx = 0
            for j, b, p_time in stage_jobs:
                order[stage_idx][machine_idx].append((j, b))
                machine_idx = (machine_idx + 1) % self.Nm[stage_idx]
        
        return order
    
    def balanced_load(self):
        """Balance load across machines"""
        order = [[] for _ in range(len(self.Nm))]
        for stage_idx in range(len(self.Nm)):
            for machine_idx in range(self.Nm[stage_idx]):
                order[stage_idx].append([])
        
        # For each stage, balance load
        for stage_idx in range(len(self.Nm)):
            # Collect all jobs with their processing times
            jobs_with_time = []
            for j in range(len(self.Nj)):
                for b in range(len(self.jobs[j])):
                    p_time = self.jobs[j][b] * self.jgsj[j, stage_idx]
                    if p_time > 0:
                        jobs_with_time.append((j, b, p_time))
            
            # Sort by processing time (descending) for better balancing
            jobs_with_time.sort(key=lambda x: x[2], reverse=True)
            
            # Track load on each machine
            machine_loads = np.array([0] * self.Nm[stage_idx])
            
            # Assign each job to machine with minimum load
            for j, b, p_time in jobs_with_time:
                min_machine = np.argmin(machine_loads)
                order[stage_idx][min_machine].append((j, b))
                machine_loads[min_machine] += p_time
        
        return order
