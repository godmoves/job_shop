import numpy as np
import matplotlib.pyplot as plt

from case_data import *


class Timer():
    def __init__(self, case_id, orders, verbose=False, random_mode=False):
        self.case_id = case_id
        self.machines = MACHINES
        self.stages = STAGES

        self.job_num = JOB_NUM[case_id]
        self.min_batch = MIN_BATCH[case_id]
        self.process_time = PROCESS_TIME[case_id]
        self.change_time = CHANGE_TIME[case_id]

        self.jobs = JOBS[case_id]
        self.max_batch_num = MAX_BATCH_NUM[case_id]

        self.work_index = WORK_INDEX[case_id]
        self.fix_index = FIX_INDEX[case_id]

        self.work_time = dict()
        self.fix_time = dict()
        self.reset_time_record()

        self.orders = orders

        # Some assertion to ensure everything is right
        assert len(self.jobs) == self.job_num
        assert all([[jj >= self.min_batch for jj in j] for j in self.jobs])
        assert self.process_time.shape == (self.job_num, self.stages - 1)
        assert self.change_time.shape == (self.job_num, self.job_num, self.stages - 1)

        self.t = np.zeros((self.job_num, self.max_batch_num, self.stages, np.max(self.machines)))
        self.c = np.zeros((self.job_num, self.max_batch_num, self.stages, np.max(self.machines)))

        self.verbose = verbose
        self.random_mode = random_mode

        self.gantt_time = []

    def get_random_delay(self, stage_id, machine_id, process_time):
        look_up = stage_id * 100 + machine_id
        if not self.random_mode:
            return 0
        if self.work_time[look_up] == 0:
            return 0
        if self.work_time[look_up] >= process_time:
            self.work_time[look_up] -= process_time
            return 0
        else:
            delay_time = self.fix_time[look_up]
            self.work_time[look_up] = self.work_index[stage_id] * -np.log(np.random.rand())
            self.fix_time[look_up] = self.fix_index[stage_id] * -np.log(np.random.rand())
            return delay_time

    def reset_time_record(self):
        for s, m_num in enumerate(self.machines):
            for m in range(m_num):
                self.work_time[s * 100 + m] = self.work_index[s] * -np.log(np.random.rand())
                self.fix_time[s * 100 + m] = self.fix_index[s] * -np.log(np.random.rand())
        self.t = np.zeros((self.job_num, self.max_batch_num, self.stages, np.max(self.machines)))

    def get_prev_stage_time(self, job, batch, prev_stage_id):
        if prev_stage_id == 0:
            return 0
        while prev_stage_id >= 0:
            for m in range(self.machines[prev_stage_id]):
                if self.c[job, batch, prev_stage_id, m] == 1:
                    return self.t[job, batch, prev_stage_id, m]
            prev_stage_id -= 1

    def get_machine_time(self, order, stage_id, machine_id):
        time = 0
        prev_job = None if order == [] else order[0][0]
        for j, b in order:
            p_time = self.jobs[j][b] * self.process_time[j, stage_id - 1]

            if p_time != 0:
                if j != prev_job:
                    time += self.change_time[prev_job, j, stage_id - 1] * 1000
                # Two constrains:
                #   1. can't process this minibatch until the previous one on this machine is done
                #   2. can't process this minibatch until it finished previous stage
                time = max(time, self.get_prev_stage_time(j, b, stage_id - 1))
                prev_job = j

            start_time = time
            time += p_time

            d_time = self.get_random_delay(stage_id, machine_id, p_time)
            time += d_time

            # Update time matrix
            if p_time != 0:
                self.t[j, b, stage_id, machine_id] = time
                if self.verbose:
                    print("job ({}, {}) start {} end {}".format(j, b, start_time, time))
                self.gantt_time.append((j, b, stage_id, machine_id, start_time, time))
            else:
                # This minibatch is not processed on this machine
                prev_t = self.get_prev_stage_time(j, b, stage_id - 1)
                self.t[j, b, stage_id, machine_id] = prev_t
        return time

    def get_stage_time(self, stage_id):
        max_t = 0
        if self.verbose:
            print("stage {}".format(stage_id))
        for m in range(self.machines[stage_id]):
            if self.verbose:
                print("machine {}".format(m))
            order = self.orders[stage_id - 1][m]
            t = self.get_machine_time(order, stage_id, m)
            max_t = max(t, max_t)
            if self.verbose:
                print("order: {} \ntime {}".format(order, t))
        if self.verbose:
            print()
        return max_t

    def get_total_machine_id(self, stage_id, machine_id):
        total_machine_id = -1
        s = 0
        while s < stage_id:
            total_machine_id += self.machines[s]
            s += 1
        total_machine_id += machine_id
        # print("stage {} machine {} tm_id {}".format(stage_id, machine_id, total_machine_id))
        return total_machine_id

    def get_total_time(self, repeat=1):
        self.reset_time_record()
        # Initialize which machine we choose at each stage
        for s in range(1, self.stages):
            for m in range(self.machines[s]):
                order = self.orders[s - 1][m]
                for j, b in order:
                    self.c[j, b, s, m] = 1

        # Calculate time information
        time_record = []
        for r in range(repeat):
            for s in range(1, self.stages):
                t = self.get_stage_time(s)
            print("Round {} Total time {:.2f}".format(r, t))
            time_record.append(t)
        summay_info = "Case {} avg time {:.2f} ({} rounds)".format(
            self.case_id, np.mean(time_record), repeat)
        print("\n{}".format(summay_info))

        if repeat == 1 and not self.random_mode:
            plt.title("Case {} gantt".format(self.case_id))
            for g in self.gantt_time:
                tm_id = self.get_total_machine_id(g[2], g[3])
                plt.barh(tm_id, g[5] - g[4], left=g[4])
            plt.ylabel("machine id")
            plt.xlabel("time")
            plt.show()

        if repeat > 10:
            plt.hist(time_record)
            plt.xlabel("time")
            plt.ylabel("number")
            plt.title(summay_info)
            plt.show()
