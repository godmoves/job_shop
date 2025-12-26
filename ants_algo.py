import numpy as np
import random
import matplotlib.pyplot as plt

from case_data import *


class Machine(object):
    def __init__(self, mid, tp, work_index=np.inf, fix_index=0):
        self.mid = mid
        self.tp = tp
        self.jid = -1
        self.ed = 0
        self.fault = False
        self.work_index = work_index
        self.fix_index = fix_index
        self.work_time = work_index * -np.log(np.random.rand())
        self.fix_time = fix_index * -np.log(np.random.rand())
        # if fix_index != 0:
        #     print("machine {} work time {} fix time {}".format(tp, self.work_time, self.fix_time))
        return

    def get_delay_time(self, process_time):
        if self.fix_index == 0:
            return 0
        if self.work_time >= process_time:
            self.work_time -= process_time
            return 0
        else:
            delay_time = self.fix_time
            # print("job delayed {} at machine {}".format(delay_time, self.mid))
            self.work_time = self.work_index * -np.log(np.random.rand())
            self.fix_time = self.fix_index * -np.log(np.random.rand())
            return delay_time


class Job(object):
    def __init__(self, jid, tp, stages, batch_size, jgsj, hmsj):
        self.batch_size = batch_size
        self.jid = jid
        self.tp = tp  # job type
        self.stages = stages
        self.si = -1
        self.jgsj = jgsj
        self.hmsj = hmsj

        self.stage_time = []  # [(ststartart, end)]
        return

    def _do(self, mid, jobs, machines):
        self.si += 1

        mx = machines[mid]

        if self.si == 0:
            last_time = 0
        else:
            last_time = self.stage_time[-1][1]

        # print ("last job time\t", last_time)
        # print ("last machine time\t", mx.ed)

        if mx.jid < 0:
            jg = self.jgsj[self.tp, mx.tp] * self.batch_size
            st = max(last_time, mx.ed)
            delay = mx.get_delay_time(jg)
            ed = st + jg + delay

            self.stage_time.append((st, ed))
            mx.ed = ed
        else:
            last_tp = jobs[mx.jid].tp
            jg = self.jgsj[self.tp, mx.tp] * self.batch_size
            hm = self.hmsj[mx.tp, last_tp, self.tp] * 1000
            delay = mx.get_delay_time(jg)

            # print ("hm time\t\t", hm)

            st = max(last_time, mx.ed + hm)
            ed = st + jg + delay

            self.stage_time.append((st, ed))
            mx.ed = ed

        mx.jid = self.jid
        return


class Ant(object):
    def __init__(self, mtp2id):
        self.path = []  # (jid, sid, mid)
        self.time = 0
        self.mtp2id = mtp2id

    def _start(self, jobs, machines, phe):
        a, b, c, d = phe.shape
        # start from super point (a-1, b-1)
        phesum = np.sum(phe[a - 1, b - 1, :c - 1, 0])
        r = random.random() * phesum
        choice = 0
        while r > 0:
            r -= phe[a - 1, b - 1, choice, 0]
            choice += 1
        choice -= 1

        self.path.append((choice, 0, 0))

        jobs[choice]._do(mid=0, jobs=jobs, machines=machines)

    def _walk_next(self, jobs, machines, phe):
        ljid, lsid, lmid = self.path[-1]

        phesum = 0
        choices = []
        for jid, job in enumerate(jobs):
            if (job.si + 1) == len(job.stages):
                continue
            tg_mx_tp = job.stages[job.si + 1]
            for mid in self.mtp2id[tg_mx_tp]:
                phesum += phe[ljid, lmid, jid, mid]
                choices.append((jid, job.si + 1, mid))

        r = random.random() * phesum
        for c, choice in enumerate(choices):
            jid, sid, mid = choice
            r -= phe[ljid, lmid, jid, mid]
            if r < 0:
                break

        jid, sid, mid = choices[c]
        self.path.append((jid, sid, mid))

        jobs[jid]._do(mid=mid, jobs=jobs, machines=machines)

    def _count_time(self, jobs):
        self.time = max(list(map(lambda job: job.stage_time[-1][1], jobs)))

    def _walk_all(self, jobs, machines, phe, stage_num):
        self._start(jobs, machines, phe)
        jid, sid, mid = self.path[-1]
        # print "job %d(%d)" % (jobs[jid].tp, jid), "stage", sid, "machine %d(%d)" % (machines[mid].tp, mid), "time", jobs[jid].stage_time[-1]
        for w in range(stage_num - 1):
            self._walk_next(jobs, machines, phe)
            jid, sid, mid = self.path[-1]
            # print "job %d(%d)" % (jobs[jid].tp, jid), "stage", sid, "machine %d(%d)" % (machines[mid].tp, mid), "time", jobs[jid].stage_time[-1]
        self._count_time(jobs)


class AntsAlgorithm():
    def __init__(self, case_id, verbose=False, random_mode=False):
        self.case_id = case_id
        self.verbose = verbose
        self.random_mode = random_mode

        # Case parameters
        self.batch_size = MIN_BATCH[case_id]
        self.Nm = MACHINES[1:]
        self.Nj = list(map(sum, JOBS[case_id]))
        self.jgsj = PROCESS_TIME[case_id]
        self.hmsj = CHANGE_TIME[case_id].transpose(2, 0, 1)
        self.gylj = self.update_gylj()
        self.mtp2id, self.machine_num = self.update_mtp2id()
        self.stage_num, self.job_num = self.update_stage_job()

        # Hyper parameters
        self.epoch_num = 300
        self.ant_per_epoch = 200
        self.lam = 2000.0
        self.ro = 0.95
        self.phe = np.ones((self.job_num + 1,
                            self.machine_num + 1,
                            self.job_num + 1,
                            self.machine_num + 1))

        # Record
        self.bstime = 9999999
        self.bsant = None

    def update_gylj(self):
        res = []
        for jg in self.jgsj:
            temp_res = []
            for stage, time in enumerate(jg):
                if time > 0:
                    temp_res.append(stage)
            res.append(temp_res)
        return res

    def update_mtp2id(self):
        machine_num = 0
        mtp2id = {}
        for tp, m in enumerate(self.Nm):
            mtp2id[tp] = []
            for i in range(m):
                mtp2id[tp].append(machine_num)
                machine_num += 1
        if self.verbose:
            print("%d machines" % machine_num)
        return mtp2id, machine_num

    def update_stage_job(self):
        stage_num = 0
        job_num = 0
        for i, n in enumerate(self.Nj):
            n_ = n
            while n_ > 0:
                job_num += 1
                stage_num += len(self.gylj[i])
                n_ -= self.batch_size
        if self.verbose:
            print("%d jobs" % job_num)
            print("%d stages" % stage_num)
        return stage_num, job_num

    def update_phe(self, ants):
        self.phe *= self.ro
        dphe = np.zeros(self.phe.shape)

        for i, ant in enumerate(ants):
            time = ant.time
            # print "ant %d time: %d" % (i, time)
            for i, p in enumerate(ant.path):
                jid, sid, mid = p

                if i == 0:
                    dphe[-1, -1, jid, mid] += self.lam / time
                else:
                    ljid, lsid, lmid = ant.path[i - 1]
                    dphe[ljid, lmid, jid, mid] += self.lam / time

        self.phe += dphe

    def get_best(self, ants):
        times = list(map(lambda ant: ant.time, ants))
        ids = range(len(ants))
        score_dict = dict(zip(times, ids))
        bstime = min(times)
        bsant = ants[score_dict[bstime]]
        return bsant, bstime

    def parse_jid(self, it):
        jid = it[0]
        c = 0
        for i, n in enumerate(self.Nj):
            n_ = n
            curr_batch = 0
            while n_ > 0:
                n_ -= self.batch_size
                if c == jid:
                    return (i, curr_batch)
                c += 1
                curr_batch += 1
        return (-1, -1)  # not find this job

    def parse_sid(self, it):
        jid = self.parse_jid(it)[0]
        sid = it[1]
        return self.gylj[jid][sid]

    def parse_mid(self, it):
        mid = it[2] + 1
        tp = 0
        curr_max_mid = self.Nm[tp]
        while mid > curr_max_mid:
            tp += 1
            curr_max_mid += self.Nm[tp]
        return mid - (curr_max_mid - self.Nm[tp]) - 1

    def parse_path(self, path):
        res1 = []
        for it in path:
            res1.append((self.parse_jid(it), self.parse_sid(it), self.parse_mid(it)))
        res2 = []
        for i, m in enumerate(self.Nm):
            temp_s = []
            for j in range(m):
                temp_m = []
                for r in res1:
                    if r[1] == i and r[2] == j:
                        temp_m.append(r[0])
                temp_s.append(temp_m)
            res2.append(temp_s)
        return res2

    def run(self, epoch_num=None):
        epoch_num = self.epoch_num if epoch_num is None else epoch_num
        time_record = []
        for epoch in range(epoch_num):
            ants = [Ant(self.mtp2id) for _ in range(self.ant_per_epoch)]
            for ant in ants:
                # create machines
                machines = []
                c = 0
                for tp, m in enumerate(self.Nm):
                    for i in range(m):
                        if self.random_mode:
                            machines.append(Machine(
                                mid=c, tp=tp,
                                work_index=WORK_INDEX[self.case_id][tp + 1],
                                fix_index=FIX_INDEX[self.case_id][tp + 1]))
                        else:
                            machines.append(Machine(mid=c, tp=tp, work_index=0, fix_index=0))
                        c += 1

                # create jobs
                jobs = []
                c = 0
                for i, n in enumerate(self.Nj):
                    n_ = n
                    while n_ > 0:
                        job = Job(jid=c, tp=i, stages=self.gylj[i],
                                  batch_size=min(self.batch_size, n_),
                                  jgsj=self.jgsj, hmsj=self.hmsj)
                        jobs.append(job)
                        n_ -= self.batch_size
                        c += 1

                # record jobs
                ant._walk_all(jobs, machines, self.phe, self.stage_num)
                ant.job_record = jobs

            self.update_phe(ants)

            bsant_, bstime_ = self.get_best(ants)
            if bstime_ < self.bstime:
                self.bstime = bstime_
                self.bsant = bsant_

            time_record.append(self.bstime)
            if epoch % 10 == 0:
                print("epoch {} best time {}".format(epoch, self.bstime))

        plt.plot(time_record)
        plt.title("case {} aa log".format(self.case_id))
        plt.xlabel("epoch")
        plt.ylabel("best time")
        plt.show()

        order = self.parse_path(self.bsant.path)
        if self.verbose:
            print("best time", self.bstime, "phe", np.sum(self.phe))
            print("best ant path: ", order)

        print(time_record)
        print(order)
        return order
