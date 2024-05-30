# from max_cut import *
from wierd_max_cut import *




import sys
import time
import os
import numpy as np
from random import sample
from random import randint, random
from math import pow, log, ceil, fabs, exp
import matplotlib.pyplot as plt

def F(x):
    return FS(x)


def Card(x):
    return x.sum()


n = get_n()
delta = 5
n_sl = 10
half = int(np.ceil(n / 2))

popu_sum = np.zeros(n, 'int8')
p_size = 1

def mutate_to_card_tar(x, k, l_b, r_b, pm=-1):
    x_ori = np.copy(x)
    cardx = Card(x)
    k_x = k - cardx
    n = len(x)
    if pm == -1:
        pm = 1 / n

    # print(x, k, l_b, r_b, pm)
    p0 = (np.abs(k_x) + k_x + pm * n) / (2 * (n - cardx))
    p1 = (np.abs(k_x) - k_x + pm * n) / (2 * cardx)
    while 1:
        x = np.copy(x_ori)
        change1_to_0 = np.random.binomial(1, p1, n)
        change1_to_0 = np.multiply(x, change1_to_0)
        change1_to_0 = 1 - change1_to_0
        x = np.multiply(x, change1_to_0)

        change0_to_1 = np.random.binomial(1, p0, n)
        change0_to_1 = np.multiply(1 - x_ori, change0_to_1)

        x += change0_to_1
        if l_b < Card(x) < r_b and (x != x_ori).any():
        # if r_bound and (s != s_ori).any():
        # if 1:
            return x

def Position(s):
    return np.where(s == 1)[0]

def cross_over_uniform(x, y):
    # return x, y
    white = np.random.binomial(1, 0.5, x.shape[0]
            )         # 1 0 0 1 1
    black = 1 - white # 0 1 1 0 0
    son =       np.multiply(x, white) + np.multiply(y, black)
    daughter =  np.multiply(x, black) + np.multiply(y, white)
    return daughter, son

def cross_over_partial(x, y):

    # return x, y
    point = np.random.randint(1, len(x))
    son = np.copy(x)
    son[point:] = y[point:]
    daughter = np.copy(y)
    daughter[point:] = x[point:]
    return daughter, son
def 按某列排序( mat, col_index):
    return mat[np.argsort(mat[:, col_index])]
def Hamming_Distance(x, slot):
    return np.abs(x - slot).sum()

def get_slot_i_of(x):
    card_x = Card(x)
    if half - card_x > n_sl:
        return -1  # card太小了不考虑
    ind = int(half - card_x)
    if ind < 0:
        print(123456789 / 0)
    return ind

for i in range(380, 410):
    print(i, int(half - i))


cur_best_f = -1
cur_best_at = None

def MyPoss4Cut(n_slots, delta=10):
    best_record = []
    t_record = []

    popu_index_tuples = [(0, 0)]  # TODO 一个list[tuples]，整个popu的每个个体对应popu_slots中的index(slot_i, 个体_i)
    # TODO 分好槽的popu，索引一个个体：[slot_i, 个体_i, :]
    popu_slots = np.array(np.zeros([n_slots, delta, n], 'int8'))
    popu_slots[0, 0] = np.random.binomial(1, 0.5, n)

    f_slots = np.array(np.zeros([n_slots, delta, 1], 'float'))
    t = 0
    T = int(ceil(n* n * 100))
    print_tn = 10000
    time0 = time.time()

    all_muts = 0
    mutation_dists_sum = 0
    mut_to_slots = np.array(np.zeros(n_slots, 'int'))
    successful_muts = 0
    useful_muts = 0
    unsuccessful_muts = 0

    file_name = str(os.path.basename(__file__))
    with open(file_name + '_result.txt', 'w') as fl:
        fl.write('')
        fl.flush()
        fl.close()

    while t < T:
        t += 2
        if t % print_tn == 0:
            print(t, ' time', time.time() - time0, 's')
            best_f = -np.inf
            best_tupl = 666666666
            for tupl in popu_index_tuples:
                fc_i = f_slots[tupl]
                if fc_i[0] > best_f:
                    best_f = fc_i[0]
                    best_tupl = tupl
            if best_tupl == 666666666:
                print(9 / 0)

            x_best = popu_slots[best_tupl]
            best_f_c = f_slots[best_tupl]

            best_record.append(best_f_c[0])
            t_record.append(t)

            with open(file_name + '_result.txt', 'a') as fl:
                fl.write(str(t) + '\n')
                for i in range(popu_slots.shape[0]):
                    for j in range(popu_slots.shape[1]):
                        # fl.write(str(popu_slots[i][j])+'\n')
                        pos = Position(popu_slots[i][j])
                        for po in pos:
                            fl.write(str(po) + '\t')
                        fl.write('\t\t\t\t\t')
                        fl.write(str(len(pos)))
                        fl.write('\n')
                    fl.write('\n')

                fl.write('\n')

                fl.write(str(f_slots) + '\n')
                fl.write(str(best_tupl) + '\n')
                fl.write(str(best_f_c) + '\n')
                fl.write('\n')
                fl.flush()
                fl.close()

            print('f, cost=', best_f_c, 'Card||=', x_best.sum(), 'popSize=', len(popu_index_tuples))
            print('last epoch unsuccessful_mutation rate', int(100 * (unsuccessful_muts / all_muts)), '%')
            print('last epoch successful_mutation rate', int(100 * (successful_muts / all_muts)), '%')
            print('last epoch useful_mutation rate', int(100 * (useful_muts / successful_muts)), '%%%%%%%%%%%%%%%')
            print('last epoch really useful_mutation rate', int(100 * (useful_muts / all_muts)), '%%%%%%%%%%%%%%%')

            print(useful_muts, successful_muts, all_muts)

            print('mutation to each slots ratio=', mut_to_slots)
            print('Avg mutation distance=', mutation_dists_sum / all_muts)
            print('----------------------------------------------')
            global cur_best_at
            if cur_best_at is not None:
                print('                                                     tar ', half - cur_best_at[0])

            mut_to_slots = np.array(np.zeros(n_slots, 'int'))
            successful_muts, useful_muts, all_muts, mutation_dists_sum, unsuccessful_muts = 0, 0, 0, 0, 0


        rand_ind = np.random.randint(0, len(popu_index_tuples))  # 随机选第几个x，几是相对于popSize而言的
        x_tuple = popu_index_tuples[rand_ind]
        x = popu_slots[x_tuple]

        rand_ind = np.random.randint(0, len(popu_index_tuples))  # 随机选第几个y，几是相对于popSize而言的
        y_tuple = popu_index_tuples[rand_ind]
        y = popu_slots[y_tuple]

        x_ori = np.copy(x)
        y_ori = np.copy(y)

        x, y = cross_over_partial(x, y)
        # x, y = cross_over_uniform(x, y)

        if x.sum() > half:
            x = 1 - x
        if y.sum() > half:
            y = 1 - y

        # global cur_best_at
        # if cur_best_at is not None:
        #     tar_card = half - cur_best_at[0]
        # else:
        #     tar_card = half
        tar_card = half

        x = mutate_to_card_tar(x, tar_card, 0, n)  # x突变
        y = mutate_to_card_tar(y, tar_card, 0, n)  # y突变
        mutation_dists_sum += (np.abs(x - x_ori).sum() + np.abs(y - y_ori).sum()) / 2

        if x.sum() > half:
            x = 1 - x
        if y.sum() > half:
            y = 1 - y

        # f_x = float(F(x))
        # f_y = float(F(y))

        x_slot_index = get_slot_i_of(x)
        y_slot_index = get_slot_i_of(y)

        all_muts += 2
        if (x_slot_index < 0 or x_slot_index >= len(popu_slots) or
                (np.any(np.all(popu_slots[x_slot_index] == x, axis=1))) or  # x 在当前slot中有孪生姐妹
                (y_slot_index < 0 or y_slot_index >= len(popu_slots)) or
                (np.any(np.all(popu_slots[y_slot_index] == y, axis=1)))
        ):
            unsuccessful_muts += 2
            continue
        mut_to_slots[x_slot_index] += 1
        successful_muts += 2


        # 把x y放进各自的slot进行比较
        put_into_popu(x, x_slot_index, f_slots[x_slot_index], popu_slots[x_slot_index], popu_index_tuples)
        put_into_popu(y, y_slot_index, f_slots[y_slot_index], popu_slots[y_slot_index], popu_index_tuples)


def update_best(fx, sl_i, pi):
    global cur_best_f
    if fx > cur_best_f:
        cur_best_f = fx
        global cur_best_at
        cur_best_at = (sl_i, pi)

def put_into_popu(x, x_sl_index, f_slot, genes, tuples):
    global popu_sum
    global p_size
    global cur_best_f
    f_x = float(F(x))

    pis = [i for i in range(delta)]
    max_f = np.max(f_slot)
    min_f = np.min(f_slot)
    if f_x < min_f:
        return False
    min_ats = np.where(f_slot == min_f)[0]
    if min_f == 0:  # 有空位
        pi = min_ats[0]
        tuples.append((x_sl_index, pi))
        f_slot[pi] = f_x
        genes[pi] = x
        popu_sum += x  # 新人放入空位，故直接加到总sum之上
        p_size += 1
        update_best(f_x, x_sl_index, pi)
        return True

    # todo  最差者唯一 （直接替换掉最差者可能导致趋同！）
    if len(min_ats) == 1:
        popu_sum -= genes[min_ats[0]]
        popu_sum += x
        f_slot[min_ats[0]] = f_x
        genes[min_ats[0]] = x
        update_best(f_x, x_sl_index, min_ats[0])
        return True

    # todo 最差者有一堆, 在min_ats
    min_ham = np.inf
    min_ham_pi = None
    temp_global_sum = popu_sum + x  # todo !!!!!!!!!短暂地把新x加入，同时pop_size + 1，然后只是为了计算总ham
    if p_size != len(tuples):
        print(p_size, len(tuples))
        print(7 / 0)
    temp_pop_n = p_size + 1
    for i in min_ats:
        old_xi = genes[i]
        ham_i = (np.multiply(temp_pop_n - temp_global_sum, old_xi) + \
                np.multiply(temp_global_sum, 1 - old_xi)).sum()

        if ham_i < min_ham:
            min_ham = ham_i
            min_ham_pi = i
    x_ham = (np.multiply(temp_pop_n - temp_global_sum, x) + \
                np.multiply(temp_global_sum, 1 - x)).sum()
    if x_ham >= min_ham:
        # 用x替换掉min ham pi处
        popu_sum -= genes[min_ham_pi]
        popu_sum += x
        f_slot[min_ham_pi] = f_x
        genes[min_ham_pi] = x
        update_best(f_x, x_sl_index, min_ham_pi)
        return True




MyPoss4Cut(n_sl, delta)























