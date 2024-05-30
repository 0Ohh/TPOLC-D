import sys
import time
import os
import numpy as np
from random import sample
from random import randint, random
from math import pow, log, ceil, fabs, exp
import matplotlib.pyplot as plt
from max_cut import *


mut_print = [True]

def setMuPT():
    mut_print[0] = True


class SUBMINLIN(object):
    def __init__(self):
        self.n = get_n()
        print(self.n)
        self.cost = np.ones(self.n, 'int')
        self.B = int(np.ceil( self.n/2 ))

    def Position(self, s):
        return np.where(s == 1)[0]

    def FS(self, s):
        x = np.array(s).reshape(1, n)
        return FS(x)[0]

    def CS(self, s):
        return np.abs(self.B - s.sum())


    def mutation_new(self, s, Tar, l_bound, r_bound, der=-1):
        # 保证期望值
        x_ori = np.copy(s)
        nn = int(s.shape[0])
        n = int(s.shape[0])
        cx = np.array(self.cost)
        if der == -1:
            der = (1 / nn) * nn * cx.min()
        a = np.dot(cx, (1 - s))
        b = np.dot(cx, s)

        if b == 0:  # 全0 gene
            s[np.random.randint(0, nn)] = 1
            a = np.dot(cx, (1 - s))
            b = np.dot(cx, s)
        if a == 0:  # 全1 gene
            s[np.random.randint(0, nn)] = 0
            a = np.dot(cx, (1 - s))
            b = np.dot(cx, s)

        B_b = Tar - b
        p0 = (abs(B_b) + der + B_b) / (2 * a)
        p1 = (abs(B_b) + der - B_b) / (2 * b)
        if p0 > 1.0:
            if mut_print[0]:
                print('fuck p0', p0)
            p0 = 1.0
        if p1 > 1.0:
            if mut_print[0]:
                print('fuck p1', p1)
            p1 = 1.0
        while 1:
            x = np.copy(x_ori)
            change1_to_0 = np.random.binomial(1, p1, n)
            change1_to_0 = np.multiply(x, change1_to_0)
            change1_to_0 = 1 - change1_to_0
            x = np.multiply(x, change1_to_0)

            change0_to_1 = np.random.binomial(1, p0, n)
            change0_to_1 = np.multiply(1 - x_ori, change0_to_1)

            x += change0_to_1

            # if l_bound < self.CS(x) < r_bound and (x != x_ori).any():
            if (x != x_ori).any():
                # if 1:
                return x


    def cross_over_uniform(self, x, y):
        # return x, y
        white = np.random.binomial(1, 0.5, x.shape[0]
                )         # 1 0 0 1 1
        black = 1 - white # 0 1 1 0 0
        son =       np.multiply(x, white) + np.multiply(y, black)
        daughter =  np.multiply(x, black) + np.multiply(y, white)
        return daughter, son


    def cross_over_partial(self, x, y):

        # return x, y

        point = np.random.randint(1, len(x))
        son = np.copy(x)
        son[point:] = y[point:]
        daughter = np.copy(y)
        daughter[point:] = x[point:]
        return daughter, son

    def Hamming_Distance(self, x, slot):
        return np.abs(x - slot).sum()

    def MyPOSS(self, B, n_slots, L, R=None, delta=5):
        if R is None: R = L
        # n_slots += (n_slots % 2)
        wd = (L + R) / n_slots
        bei = L // wd
        wd = L / bei
        R = wd * n_slots - L

        # c = np.array(self.cost)
        # c = np.sort(c)
        # print(c)

        best_record = []
        t_record = []

        print(self.n)
        print(self.cost)
        print(B)
        if R is None: R = L
        print(B-L, B, B+R)

        popu_index_tuples = [(0, 0)]  # TODO 一个list[tuples]，整个popu的每个个体对应popu_slots中的index(slot_i, 个体_i)
                                # TODO 这个列表只会append，不会删东西
        # popSize = 1
        # TODO 分好槽的popu，索引一个个体：[slot_i, 个体_i, :]
        popu_slots = np.array(np.zeros([n_slots, delta, self.n], 'int8'))
        # TODO 分好槽的popu的f，cost； 索引一个个体的f / c：[slot_i, 个体_i, (0 / 1)]
        f_c_slots = np.array(np.zeros([n_slots, delta, 2], 'float'))
        ham_slots = np.array(np.zeros([n_slots, delta, 1], 'float'))
        slot_wid = (L + R) / n_slots

        t = 0
        T = int(ceil(self.n * self.n * 100))
        print_tn = 10000
        time0 = time.time()

        all_muts,mutation_dists_sum,successful_muts,useful_muts,unsuccessful_muts = 0,0,0,0,0
        mut_to_slots = np.array(np.zeros(n_slots, 'int'))

        file_name = str(os.path.basename(__file__))
        with open(file_name + '_result.txt', 'w') as fl:
            fl.write('')
            fl.flush()
            fl.close()

        while t < T:
            self.epoch = t
            t += 2
            if t % print_tn == 0:
                print(t, ' time', time.time() - time0, 's')
                best_f = -np.inf
                best_tupl = 666666666
                for tupl in popu_index_tuples:
                    fc_i = f_c_slots[tupl]
                    if fc_i[1] > B:
                        continue
                    if fc_i[0] > best_f:
                        best_f = fc_i[0]
                        best_tupl = tupl
                if best_tupl == 666666666:
                    print(9 / 0)

                x_best = popu_slots[best_tupl]
                best_f_c = f_c_slots[best_tupl]

                best_record.append(best_f_c[0])
                t_record.append(t)

                with open(file_name+'_result.txt', 'a') as fl:
                    fl.write(str(t) + '\n')
                    for i in range(popu_slots.shape[0]):
                        for j in range(popu_slots.shape[1]):
                            # fl.write(str(popu_slots[i][j])+'\n')
                            pos = self.Position(popu_slots[i][j])
                            for po in pos:
                                fl.write(str(po)+'\t')
                            fl.write('\t\t\t\t\t')
                            fl.write(str(len(pos)))
                            fl.write('\n')
                        fl.write('\n')

                    fl.write('\n')

                    fl.write(str(f_c_slots) + '\n')
                    fl.write(str(best_tupl) + '\n')
                    fl.write(str(best_f_c) + '\n')
                    fl.write('\n')
                    fl.flush()
                    fl.close()

                print('f, cost=',  best_f_c, 'Card||=', x_best.sum(), 'popSize=', len(popu_index_tuples))
                print('last epoch unsuccessful_mutation rate', int(100*(unsuccessful_muts/all_muts)), '%')
                print('last epoch successful_mutation rate', int(100*(successful_muts/all_muts)), '%')
                print('last epoch useful_mutation rate', int(100*(useful_muts/successful_muts)), '%%%%%%%%%%%%%%%')
                print('last epoch really useful_mutation rate', int(100*(useful_muts/all_muts)), '%%%%%%%%%%%%%%%')

                print(useful_muts, successful_muts, all_muts)

                print('mutation to each slots ratio=', mut_to_slots)
                print('Avg mutation distance=', mutation_dists_sum/all_muts)
                print('----------------------------------------------')

                mut_to_slots = np.array(np.zeros(n_slots, 'int'))
                successful_muts, useful_muts, all_muts, mutation_dists_sum, unsuccessful_muts = 0, 0, 0, 0, 0
                if t > 5*print_tn:
                    setMuPT()

            rand_ind = np.random.randint(0, len(popu_index_tuples))  # 随机选第几个x，几是相对于popSize而言的
            x_tuple = popu_index_tuples[rand_ind]
            x = popu_slots[x_tuple]

            rand_ind = np.random.randint(0, len(popu_index_tuples))  # 随机选第几个y，几是相对于popSize而言的
            y_tuple = popu_index_tuples[rand_ind]
            y = popu_slots[y_tuple]

            x_ori = np.copy(x)

            x, y = self.cross_over_partial(x, y)
            # x, y = self.cross_over_uniform(x, y)

            if x.sum() > B:
                x = 1 - x
            if y.sum() > B:
                y = 1 - y

            x = self.mutation_new(x, (B-L+B+R)/2, B-L, B+R)  # x突变
            y = self.mutation_new(y, (B-L+B+R)/2, B-L, B+R)  # y突变

            if x.sum() > B:
                x = 1 - x
            if y.sum() > B:
                y = 1 - y


            mutation_dists_sum += np.abs(x - x_ori).sum()

            f_x = float(self.FS(x))  # todo 把hamming distance考量纳入到 cost中？ 还是说f？还是按cowding dist？？
            cost_x = self.CS(x)
            f_y = float(self.FS(y))
            cost_y = self.CS(y)

            x_slot_index = int(
                np.ceil((cost_x -0.00000000001  - (B - L)) / slot_wid)  # 向上取整除法
            ) - 1
            y_slot_index = int(
                np.ceil((cost_y -0.00000000001 - (B - L)) / slot_wid)  # 向上取整除法
            ) - 1

            all_muts += 1
            if x_slot_index < 0 or x_slot_index >= len(popu_slots):
                unsuccessful_muts += 1
                continue
            if np.any(np.all(popu_slots[x_slot_index] == x, axis=1)):  # x 在当前slot中有孪生姐妹
                unsuccessful_muts += 1
                continue
            if y_slot_index < 0 or y_slot_index >= len(popu_slots):
                unsuccessful_muts += 1
                continue
            if np.any(np.all(popu_slots[y_slot_index] == y, axis=1)):  # x 在当前slot中有孪生姐妹
                unsuccessful_muts += 1
                continue

            mut_to_slots[x_slot_index] += 1
            successful_muts += 1

            # cost_x = cost_x * (self.n - self.Hamming_Distance(x, popu_slots[x_slot_index]))
            # cost_y = cost_y * (self.n - self.Hamming_Distance(y, popu_slots[y_slot_index]))

            self.put_into_popu(x, x_slot_index, f_x, cost_x, popu_slots, f_c_slots, ham_slots, popu_index_tuples, delta)
            self.put_into_popu(y, y_slot_index, f_y, cost_y, popu_slots, f_c_slots, ham_slots, popu_index_tuples, delta)

        # end While
        # 输出答案
        best_f = -np.inf
        best_tupl = 666666666
        for tupl in popu_index_tuples:
            fc_i = f_c_slots[tupl]
            if fc_i[1] > B:
                continue
            if fc_i[0] > best_f:
                best_f = fc_i[0]
                best_tupl = tupl
        x_best = popu_slots[best_tupl]
        best_f_c = f_c_slots[best_tupl]
        return x_best, best_f_c

    def put_into_popu(self, x, x_slot_index, f_x, cost_x, popu_slots, f_c_slots, ham_slots, popu_index_tuples, delta):
        # TODO 把x与slot内所有个体比较 f，（可能需要维护slot全体f,c值的np array）
        worst_x_index = None
        worst_f = np.inf
        worst_cost = -1.0
        x_is_added = False
        for p in range(0, delta):  # p-> (0~5)
            if f_c_slots[x_slot_index, p, 1] == 0:  # (第p个)某旧个体cost==0，即全0gene,说明槽未满
                # 直接把x放进这里
                x_is_added = True
                popu_slots[x_slot_index, p] = x
                f_c_slots[x_slot_index, p, 0] = f_x
                f_c_slots[x_slot_index, p, 1] = cost_x
                popu_index_tuples.append((x_slot_index, p))
                break
            if (f_c_slots[x_slot_index, p, 0] < worst_f
                    or
                    (f_c_slots[x_slot_index, p, 0] == worst_f and f_c_slots[x_slot_index, p, 1] >= worst_cost)
            ):
                # 遍历的第p比当前两方面最差的个体还要 更差（或一样差）
                worst_f = f_c_slots[x_slot_index, p, 0]
                worst_cost = f_c_slots[x_slot_index, p, 1]
                worst_x_index = p
        if (not x_is_added) and f_x > worst_f:  # x暂未加入，故当前槽已满，但新个体fx > 最差者的f
            # x替换最差者
            x_is_added = True
            popu_slots[x_slot_index, worst_x_index] = x
            f_c_slots[x_slot_index, worst_x_index, 0] = f_x
            f_c_slots[x_slot_index, worst_x_index, 1] = cost_x

        if (not x_is_added) and f_x == worst_f and cost_x < worst_cost:
            worst_ham = np.inf
            worst_ham_index = -1
            slot = popu_slots[x_slot_index]
            x_ham = self.Hamming_Distance(x, slot)
            for p in range(0, delta):
                ham_p = ham_slots[x_slot_index, p] + self.Hamming_Distance(slot[p], x)
                # print(ham_slots[x_slot_index, p], self.Hamming_Distance(slot[p], x))
                if ham_p < x_ham and ham_p < worst_ham:
                    worst_ham_index = p

            if worst_ham_index != -1:
                x_is_added = True
                popu_slots[x_slot_index, worst_ham_index] = x
                f_c_slots[x_slot_index, worst_ham_index, 0] = f_x
                f_c_slots[x_slot_index, worst_ham_index, 1] = cost_x

        if x_is_added:
            # update ham slots
            for p in range(0, delta):
                ham_slots[x_slot_index, p] = self.Hamming_Distance(popu_slots[x_slot_index, p], popu_slots[x_slot_index])

        # if (not x_is_added) and f_x == worst_f and cost_x < worst_cost:
        #     # print('hhhhhhhhhhhhhhh')
        #     x_is_added = True
        #     popu_slots[x_slot_index, worst_x_index] = x
        #     f_c_slots[x_slot_index, worst_x_index, 0] = f_x
        #     f_c_slots[x_slot_index, worst_x_index, 1] = cost_x


if __name__ == "__main__":
    myObject = SUBMINLIN()
    n = myObject.n
    B= int(np.ceil( n/2 ))
    n_sl = 40

    # myObject.MyPOSS(B, n_sl, 16, 16, delta=5)
    myObject.MyPOSS(B, n_sl, n_sl//2, n_sl//2, delta=10)
