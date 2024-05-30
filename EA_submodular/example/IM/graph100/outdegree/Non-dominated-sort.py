import sys
import time
import os
import datetime
import numpy as np
from random import sample
from random import randint, random
from math import pow, log, ceil, fabs, exp
import matplotlib.pyplot as plt

mut_print = [True]

def setMuPT():
    mut_print[0] = True



class ObjectiveIM(object):
    def __init__(self, weightMatrix,nodeNum):
        self.weightMatrix = weightMatrix
        self.n=nodeNum
        self.solution = []
        self.allNodes=np.ones((1, self.n))
        self.cost = [0] * self.n
        #eps=abs(np.random.normal(0,0.5,size=self.n))
        dataFile=open('graph100_eps.txt')
        dataLine=dataFile.readlines()
        items=dataLine[0].split()
        eps=[0]*self.n
        for i in range(self.n):
            eps[i]=float(items[i])
        dataFile.close()
        for i in range(self.n):
            #tempValue=random() #rondom cost between 0 and 1
            outDegree=(self.weightMatrix[i,:]>0).sum()
            self.cost[i]=1.0+(1+abs(eps[i]))*outDegree
        #print('size of node is %d' % (self.n))
        self.cost = np.array(self.cost)


    def FinalActiveNodes(self):  # solution is the numpy matrix 1*n
        activeNodes = np.zeros((1, self.n)) + self.solution
        cActive = np.zeros((1, self.n)) + self.solution# currently active nodes
        tempNum = int(cActive.sum(axis=1)[0, 0])
        while tempNum > 0:
            nActive = self.allNodes - activeNodes
            randMatirx = np.random.rand(tempNum, self.n)#uniformly random matrix between 0 and 1
            z = sum(randMatirx < self.weightMatrix[cActive.nonzero()[-1], :]) > 0 #cActive.nonzero()[-1] is the nonzero index
            cActive = np.multiply(nActive, z) #sum is the sum of each column,the new active node
            activeNodes = (cActive + activeNodes) > 0
            tempNum = int(cActive.sum(axis=1)[0, 0])
        return activeNodes.sum(axis=1)[0, 0]



    def EstimateObjective_accurate(self, solution):  # simulate 10000 times
        temp = np.mat(solution)
        self.solution = temp
        val = 0
        for i in range(10000):
            val += self.FinalActiveNodes()
        return val / 10000.0

    def Position(self, s):
        return np.where(s == 1)[0]

    def FS(self, solution):  # simulate 50 times
        temp = np.mat(solution)
        self.solution = temp
        val = 0
        for i in range(100):
            val += self.FinalActiveNodes()
        return val / 100.0

    def CS(self, s):
        tempSum=0
        pos = self.Position(s)
        for item in pos:
            tempSum=tempSum+self.cost[item]
        return tempSum


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
        # if p0 > 1.0:
        #     if mut_print[0]:
        #         print('fuck p0', p0)
        #     p0 = 1.0
        # if p1 > 1.0:
        #     if mut_print[0]:
        #         print('fuck p1', p1)
        #     p1 = 1.0
        while 1:
            x = np.copy(x_ori)
            change1_to_0 = np.random.binomial(1, p1, n)
            change1_to_0 = np.multiply(x, change1_to_0)
            change1_to_0 = 1 - change1_to_0
            x = np.multiply(x, change1_to_0)

            change0_to_1 = np.random.binomial(1, p0, n)
            change0_to_1 = np.multiply(1 - x_ori, change0_to_1)

            x += change0_to_1

            if l_bound < self.CS(x) < r_bound and (x != x_ori).any():
            # if (x != x_ori).any():
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

        slot_wid = (L + R) / n_slots

        self.f_leag_best_at_tuple = None
        self.f_leag_best = 0.0

        best_record = []
        t_record = []
        self.Bud = B
        print(self.n)
        print(self.cost)
        print(B)
        print(B-L, B, B+R)

        popu_index_tuples = [(0, 0)]  # TODO 一个list[tuples]，整个popu的每个个体对应popu_slots中的index(slot_i, 个体_i)
            # TODO 分好槽的popu，索引一个个体：[slot_i, 个体_i, :]
        popu_slots = np.array(np.zeros([n_slots, delta, self.n], 'int8'))
            # TODO 分好槽的popu的f，cost； 索引一个个体的f / c：[slot_i, 个体_i, (0 / 1)]
        f_c_slots = np.array(np.zeros([n_slots, delta, 2], 'float'))
        t = 0
        T = int(ceil(self.n * self.n * 10))
        print_tn = 2000
        time0 = time.time()

        all_muts = 0
        mutation_dists_sum = 0
        mut_to_slots = np.array(np.zeros(n_slots, 'int'))
        successful_muts = 0
        useful_muts = 0
        unsuccessful_muts = 0

        file_name = str(os.path.basename(__file__))
        # with open(file_name + '_result.txt', 'w') as fl:
        #     fl.write('')
        #     fl.flush()
        #     fl.close()

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
                    print(f_c_slots)
                    # return

                x_best = popu_slots[best_tupl]
                best_f_c = f_c_slots[best_tupl]

                best_real = self.EstimateObjective_accurate(popu_slots[best_tupl])

                best_record.append(best_real)
                t_record.append(t)

                with open(file_name+'_result.txt', 'a') as fl:
                    fl.write(str(t) + '\n')

                    # for i in range(popu_slots.shape[0]):
                    #     sor = self.按某列排序(f_c_slots[i], 1)
                    #     fl.write(str(sor) + '\n')
                    # fl.write(str(best_tupl) + '\n')
                    fl.write(str(best_real) + '\n')

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
                print('best_real                            ', best_real)
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
            y_ori = np.copy(y)

            # x, y = self.cross_over_partial(x, y)
            x, y = self.cross_over_uniform(x, y)

            x = self.mutation_new(x, (B-L+B+R)/2, B-L, B+R)  # x突变
            y = self.mutation_new(y, (B-L+B+R)/2, B-L, B+R)  # y突变
            mutation_dists_sum += (np.abs(x - x_ori).sum() + np.abs(y - y_ori).sum())/2

            f_x = float(self.FS(x))
            cost_x = self.CS(x)
            f_y = float(self.FS(y))
            cost_y = self.CS(y)

            x_slot_index = int(np.ceil((cost_x - (B - L)) / slot_wid)  # 向上取整除法
            ) - 1
            y_slot_index = int(np.ceil((cost_y - (B - L)) / slot_wid)  # 向上取整除法
            ) - 1

            all_muts += 2
            if (x_slot_index < 0 or x_slot_index >= len(popu_slots) or
                    (np.any(np.all(popu_slots[x_slot_index] == x, axis=1))) or     # x 在当前slot中有孪生姐妹
                    (y_slot_index < 0 or y_slot_index >= len(popu_slots)) or
                    (np.any(np.all(popu_slots[y_slot_index] == y, axis=1)))
            ):
                unsuccessful_muts += 2
                continue
            mut_to_slots[x_slot_index] += 1
            successful_muts += 2

            x_useful = self.put_into_popu_NSGA_II(x, x_slot_index, f_x, cost_x, popu_slots, f_c_slots, popu_index_tuples, delta)
            y_useful = self.put_into_popu_NSGA_II(y, y_slot_index, f_y, cost_y, popu_slots, f_c_slots, popu_index_tuples, delta)
            if x_useful:
                useful_muts += 1
            if y_useful:
                useful_muts += 1
            if x_useful or y_useful or 1:
                for si in range(popu_slots.shape[0]):
                    for pi in range(popu_slots.shape[1]):
                        if f_c_slots[si][pi][0] >= self.f_leag_best and f_c_slots[si][pi][1] <= self.Bud:
                            self.f_leag_best = f_c_slots[si][pi][0]
                            self.f_leag_best_at_tuple = (si, pi)
            # if x_useful or y_useful:
            #     # todo minor trick
            #     for si in range(1, popu_slots.shape[0]):
            #         fs = f_c_slots[si, :, 0]
            #         last_fs = f_c_slots[si - 1, :, 0]
            #         now_worst_f_at = np.argmin(fs)
            #         last_best_f_at = np.argmax(last_fs)
            #         if last_best_f_at > now_worst_f_at:
            #             # 用上一个slot的最好的f，替换掉本slot最差f的
            #             popu_slots[si, now_worst_f_at] = np.copy(
            #                 popu_slots[si - 1, last_best_f_at]
            #             )
            #             f_c_slots[si, now_worst_f_at] = np.copy(
            #                 f_c_slots[si - 1, last_best_f_at]
            #             )

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

    def DELETE_at(self, slot, slot_f, slot_c, pi):
        slot[pi] = 0.0
        slot_f[pi] = 0.0
        slot_c[pi] = 0.0

    def PUT_x_at(self, PUT_info, pi):
        x, f_x, cost_x, slot, slot_f, slot_c = PUT_info
        slot[pi] = x
        slot_f[pi] = f_x
        slot_c[pi] = cost_x

    def put_into_popu_NSGA_II(self, x, x_slot_index, f_x, cost_x, popu_slots, f_c_slots, popu_index_tuples, delta):
        genes = popu_slots[x_slot_index]
        f = f_c_slots[x_slot_index, :, 0]
        c = f_c_slots[x_slot_index, :, 1]
        PUT_info = (x, f_x, cost_x, genes, f, c)

        # TODO 当slot中有空格时，直接放x
        blank_at = np.where(c == 0.0)[0]
        if len(blank_at) > 0:
            blank_pi = blank_at[0]
            self.PUT_x_at(PUT_info, blank_pi)
            popu_index_tuples.append((x_slot_index, blank_pi))
            return True

        # TODO 把x放入f, c后，非支配排序
        f_all = np.hstack((f, [f_x]))
        c_all = np.hstack((c, [cost_x]))
        i_all = np.arange(delta + 1)  # x 加入了
        genes = np.vstack((genes, x))
        x_fake_pi = i_all[-1]   # 应该是delta

        death_pi = self.select_die_pi(i_all, f_all, c_all, genes, x_slot_index)

        if death_pi == x_fake_pi:
            return False
        else:
            self.PUT_x_at(PUT_info, death_pi)
            return True

    def select_die_pi(self, i_all, f_all, c_all, genes, x_slot_index):
        ranks, biggest_rank = self.obtain_ranks(i_all, f_all, c_all)
        if biggest_rank > 0:
            i_F_C_D = np.vstack((
                i_all,  # 列0,1,2,3,4...(包括x)
                f_all,
                c_all,
                np.zeros(len(i_all), 'float')  # todo 加一个空列，放dist
            )).T
            i_R = np.vstack((
                i_all,  # 列0,1,2,3,4...(包括x)
                ranks
            )).T
            # todo 先按 R 升序排一下
            i_R_sortR = self.按某列排序(i_R, 1)
            worst_rak = i_R_sortR[-1, 1]
            if worst_rak != i_R_sortR[-2, 1]:  # 最后一行的R（最烂的rank) 没有并列的烂R
                # 直接把 最烂R 的个体（pi）剔除
                pi = i_R_sortR[-1, 0]
                death_pi = pi
            else:  # todo  需要把最烂R并列几人，在该Rank之中算distance，剔除distance最小者
                where = np.where(i_R_sortR[:, 1] == worst_rak)[0]
                worst_pi = i_R_sortR[where, 0]
                wor_num = len(worst_pi)
                # todo  最烂R中有两个人
                if wor_num == 2:
                    a, b = worst_pi[0], worst_pi[1]
                    if f_all[a] > f_all[b] or (f_all[a] == f_all[b] and c_all[a] <= c_all[b]):
                        death_pi = b
                    else:
                        death_pi = a
                    return death_pi

                i_F_C_D = i_F_C_D[worst_pi]
                i_F_C_D_sortF = self.按某列排序(i_F_C_D, 1)
                i_F_C_D_sortC = self.按某列排序(i_F_C_D, 2)
                i_F_C_D_sortF[-1, -1] = np.inf  # max F
                i_F_C_D_sortC[0, -1] = np.inf  # min C
                for i in range(1, wor_num - 1):
                    i_F_C_D_sortF[i, -1] += np.abs(i_F_C_D_sortF[i - 1, 1] - i_F_C_D_sortF[i + 1, 1])
                for i in range(1, wor_num - 1):
                    i_F_C_D_sortC[i, -1] += np.abs(i_F_C_D_sortC[i - 1, 2] - i_F_C_D_sortC[i + 1, 2])
                # todo 按头号升序排序
                i_F_C_D1 = self.按某列排序(i_F_C_D_sortF, 0)
                i_F_C_D2 = self.按某列排序(i_F_C_D_sortC, 0)
                i_F_C_D1[:, -1] += i_F_C_D2[:, -1]  # 合并两结果（dist
                i_F_C_D_final = self.按某列排序(i_F_C_D1, -1)  # 按最后一列dist排序，最上面一个为最小dist的解！！！排除它！

                if self.epoch > 20_0000:
                    if i_F_C_D_final[0, -1] == i_F_C_D_final[1, -1]:
                        # 两个最小的distance并列了
                        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                        print(x_slot_index)
                        print(f_all)
                        print(c_all)
                        print(i_F_C_D_final)

                for i in range(0, i_F_C_D_final.shape[0]):
                    die_pi = int(i_F_C_D_final[i, 0])
                    # if self.f_leag_best_at_tuple != (x_slot_index, die_pi):  # 不会选到f_leag_best_at_tuple情况下的最小dist解
                    if 1:  # 不会选到f_leag_best_at_tuple情况下的最小dist解
                        death_pi = die_pi
                        break
            return death_pi
        else:
            # todo  总共只有一层rank=0
            # todo 尝试Hamming
            hams = []
            for _ in range(len(i_all)):
                pi = i_all[_]
                hams.append(self.Hamming_Distance(genes[pi], genes))
            # todo 必保住两个端点
            f_max = np.max(f_all)
            c_min = np.min(c_all)
            f_max_ids = np.where(f_all == f_max)[0]
            c_min_ids = np.where(c_all == c_min)[0]
            if len(c_min_ids) == 1:
                hams[c_min_ids[0]] = np.inf
            if len(f_max_ids) == 1:
                hams[f_max_ids[0]] = np.inf

            min_ham = np.inf
            die_pi = None
            for _ in range(len(i_all)):
                pi = i_all[_]
                # if self.f_leag_best_at_tuple == (x_slot_index, die_pi):   # 不会选到f_leag_best_at_tuple情况下的最小dist解
                # print((x_slot_index, pi), end=' ')
                if self.f_leag_best_at_tuple == (x_slot_index, pi):  # 不会选到f_leag_best_at_tuple情况下的最小dist解
                    continue
                if hams[_] < min_ham:
                    min_ham = hams[_]
                    die_pi = pi
            return die_pi



    def obtain_ranks(self, i_all, f_all, c_all):
        r = 0
        not_done = True
        domed_nums_of = np.zeros(len(i_all), 'int')  # 0, 0, 0, ...
        ranks = np.zeros(len(i_all), 'int')
        ranks += (len(i_all) + 6)  # 大，大， 大， 大... (rank 最大只能=delta)
        biggest_rank = -1
        while not_done:
            not_done = False
            domed_nums_of = np.zeros(len(i_all), 'int')
            for i in i_all:
                if ranks[i] <= r - 1:  # 若i为更高级支配层的 就不计入i的任何支配
                    continue
                for j in i_all:  # 让所有被i支配的j，受支配数+1
                    if j == i: continue
                    if ((f_all[i] > f_all[j] and c_all[i] <= c_all[j]) or
                            (f_all[i] >= f_all[j] and c_all[i] < c_all[j])
                    ):
                        domed_nums_of[j] += 1

            pis_of_r = np.where(domed_nums_of == 0)[0]
            for pi in pis_of_r:  # domed数量为0的pi们：
                if ranks[pi] <= r - 1:  # domed数量为0的这个pi，是之前统计过r的nb个体
                    continue
                if ranks[pi] != (len(i_all) + 6):
                    print(99 / 0)
                else:
                    not_done = True
                    ranks[pi] = r
                    if r > biggest_rank:
                        biggest_rank = r
            r += 1
        return ranks, biggest_rank

    def 按某列排序(self, mat, col_index):
        return mat[np.argsort(mat[:, col_index])]


def ReadData(p,filePath):
    dataFile=open(filePath)
    maxNode=0
    while True:
        line=dataFile.readline()
        if not line:
            break
        items=line.split()
        if len(items)>0:
            start=int(items[0])
            end=int(items[1])
            if start>maxNode:
                maxNode=start
            if end>maxNode:
                maxNode=end
    dataFile.close()
    maxNode=maxNode
    print(maxNode)
    data = np.mat(np.zeros([maxNode, maxNode]))
    dataFile = open(filePath)
    while True:
        line = dataFile.readline()
        if not line:
            break
        items = line.split()
        if len(items)>0:
            data[int(items[0])-1,int(items[1])-1]=p
    dataFile.close()
    return data


if __name__ == "__main__":

    for pp in range(20):

        p=0.05
        filePath='graph100-01.txt'
        B=50
        weightMatrix=ReadData(p,filePath)
        nodeNum=np.shape(weightMatrix)[0]
        myObject=ObjectiveIM(weightMatrix,nodeNum)
        B_ = 100
        n_sl = 8

        coo = np.array(myObject.cost)

        myObject.MyPOSS(B_, n_sl, 10, 10, delta=5)

