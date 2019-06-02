import numpy as np

from job_shop import cal_scores


class GeneticAlgo:
    def __init__(self, pop_size, chrom_len, machines, jobs, max_priority=100):
        self.pop_size = pop_size
        self.chrom_len = chrom_len
        self.max_priority = max_priority
        self.mutate_rate = 0.1
        self.cross_rate = 0.1
        self.machines = machines
        self.jobs = jobs
        self.keep_best_num = 3
        self.tournament_size = int(0.1 * pop_size)

    def init(self):
        assert self.chrom_len == 2 * (len(self.machines) - 1) * sum(map(len, self.jobs))
        stage_jobs = sum(map(len, self.jobs))
        self.pop = np.zeros((self.pop_size, self.chrom_len))
        for i in range(self.pop_size):  # each chromosome
            for j in range(len(self.machines) - 1):  # each stage
                for k in range(stage_jobs):  # each minibatch
                    idx = j * stage_jobs + k
                    if idx % 2 == 1:
                        self.pop[i, idx] = np.random.randint(0, self.max_priority)
                    else:
                        self.pop[i, idx] = np.randon.randint(0, self.machines[j])

    def mutate(self, pop_id):
        rate = np.random.rand()
        if rate > self.mutate_rate:
            return self.pop[pop_id, :]

        pos = np.random.randint(0, self.chrom_len)
        # chrom = self.pop[pop_id, :]
        # if pos % 2 == 1:
        #     chrom[pos] = np.random.randint(0, self.max_priority)
        # else:
        #     chrom[pos] = np.randon.randint(0, self.machines[pos // 2])
        # return chrom
        if pos % 2 == 1:
            self.pop[pop_id, pos] = np.random.randint(0, self.max_priority)
        else:
            self.pop[pop_id, pos] = np.randon.randint(0, self.machines[pos // 2])

    def cross(self, father_id):
        rate = np.random.rand()
        if rate > self.cross_rate:
            return self.pop[father_id, :]

        mather_id = np.random.randint(0, self.pop_size)
        if father_id == mather_id:
            return self.pop[father_id, :]

        pos = np.random.randint(0, self.chrom_len)
        # child = self.pop[father_id, :]
        # child[pos:] = self.pop[mather_id, :][pos:]
        # return child
        self.pop[father_id, pos:] = self.pop[mather_id, pos:]

    def select(self):
        new_pop = np.zeros((self.pop_size, self.chrom_len))
        scores = cal_scores(self.pop)

        for i in range(self.pop_size):
            best_score = np.inf
            best_id = 0
            for j in range(self.tournament_size):
                pop_id = np.random.randint(0, self.pop_size)
                if scores[pop_id] < best_score:
                    best_id = pop_id
                    best_score = scores[pop_id]
            new_pop[i, :] = self.pop[best_id, :]
        self.pop = new_pop

        return max(scores)

    def run(self, epoch_num):
        for i in range(epoch_num):
            print('epoch: {}'.format(i))
            for j in range(self.pop_size):
                self.mutate(j)
                self.cross(j)
            best_score = self.select()
            print('best score: {}'.format(best_score))
