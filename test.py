from orders import ORDERS
from timer import Timer
from ants_algo import AntsAlgorithm


ID = 3

# Task 1 & 2
aa = AntsAlgorithm(case_id=ID, verbose=False, random_mode=False)
aa_order = aa.run(epoch_num=500)
aa_timer = Timer(case_id=ID, orders=aa_order, random_mode=False)
aa_timer.get_total_time(repeat=1)

# # Task 2
# optimal_timer = Timer(case_id=ID, orders=ORDERS[ID], random_mode=True)
# optimal_timer.get_total_time(repeat=1000)

# # Task 1
# optimal_timer = Timer(case_id=ID, orders=ORDERS[ID], random_mode=False)
# optimal_timer.get_total_time(repeat=1)
