import os
import sys
import time
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
        self.cost=[0]*self.n
        dataFile=open('graph100_rand_cost.txt')
        dataLine=dataFile.readlines()
        items=dataLine[0].split()
        for i in range(self.n):
            tempValue=float(items[i])
            if tempValue>0:
                self.cost[i]=tempValue
        self.cost = np.array(self.cost)
        dataFile.close()
        #print('size of node is %d' % (self.n))



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

    def CS(self,s):
        tempSum=0
        pos=self.Position(s)
        for item in pos:
            tempSum=tempSum+self.cost[item]
        return tempSum



    def mutation_new(self, s, Tar, l_bound, r_bound, der=-1, prit=True):
        # 保证期望值


        # 设置Tar为：输入Tar（大概是B）与当前cost的平均值 -> 变化不大，时好时坏，但好像是会好一点
        # Tar = (self.CS(s) + Tar) / 2

        # 设置Tar为： 当前cost关于输入Tar的镜像    -> 收敛更快，但似乎容易卡主，早衰严重（即种群退化
        # Tar = Tar + Tar - self.CS(s)


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

        if prit:
            print(p0*(1-s).sum(), p1*s.sum(), ' ---------------------------> p0*n0, p1*n1')
            print(p0*nn, p1*nn, ' -----------~~~~~~~~~~~~~~~~~~~~~~~~~~> p0*n, p1*n')

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

            if r_bound < self.CS(x):
                if prit:
                    print('cost太大了，1太多了')
            if self.CS(x) < l_bound:
                if prit:
                    print('太小了')

            if l_bound < self.CS(x) < r_bound and (x != x_ori).any():
            # if (x != x_ori).any():
                # if 1:
                return x

    def cross_over_2_part(self, x, y):
        _ = np.random.randint(0, len(x))
        __ = np.random.randint(0, len(x))
        p1 = min(_, __)
        p2 = max(_, __)
        son = np.copy(x)
        son[p1:p2] = y[p1:p2]
        daughter = np.copy(y)
        daughter[p1:p2] = x[p1:p2]
        return daughter, son


    def cross_over_partial(self, x, y):

        # return x, y

        point = np.random.randint(1, len(x))
        son = np.copy(x)
        son[point:] = y[point:]
        daughter = np.copy(y)
        daughter[point:] = x[point:]
        return daughter, son

    def cross_over_uniform(self, x, y):
        # return x, y
        white = np.random.binomial(1, 0.5, x.shape[0]
                )         # 1 0 0 1 1
        black = 1 - white # 0 1 1 0 0
        son =       np.multiply(x, white) + np.multiply(y, black)
        daughter =  np.multiply(x, black) + np.multiply(y, white)
        return daughter, son



    def MyPOSS(self, B, n_slots, L, R=None, delta=5):
        if R is None: R = L
        # n_slots += (n_slots % 2)
        wd = (L + R) / n_slots
        bei = L // wd
        wd = L / bei
        R = wd * n_slots - L
        slot_wid = (L + R) / n_slots

        best_record = []
        t_record = []

        print(self.n)
        print(self.cost)
        print(B)
        if R is None: R = L
        print(B-L, B, B+R)

        inital = np.zeros(self.n, 'int')
        init_c = 0.0
        ci = np.array([_ for _ in range(len(self.cost))])
        ic = np.vstack((ci, self.cost))


        ic = ic[:, np.argsort(ic[1, :])]
        for i in range(self.n - 1, -1, -1):
            cur_cost = ic[1, i]
            if init_c + cur_cost <= B:
                # add 1s on place ic[0, i]
                inital[int(ic[0, i])] = 1
                init_c += cur_cost
        print(inital, init_c, 'initntintintinininintinininitnininini')
        init_sl_i = int(
                np.ceil((init_c - (B - L)) / slot_wid)  # 向上取整除法
            ) - 1


        self.global_sum = np.zeros(self.n, 'int')
        # self.global_sum = np.copy(inital)
        self.popu_index_tuples = [(0, 0)]  # TODO 一个list[tuples]，整个popu的每个个体对应popu_slots中的index(slot_i, 个体_i)
        # self.popu_index_tuples = [(init_sl_i, 0)]  # TODO 一个list[tuples]，整个popu的每个个体对应popu_slots中的index(slot_i, 个体_i)


        # self.popu_index_tuples = []
        # for i in range(n_slots):
        #     for j in range(delta):
        #         self.popu_index_tuples.append((i, j))


        # popSize = 1
        # TODO 分好槽的popu，索引一个个体：[slot_i, 个体_i, :]
        popu_slots = np.array(np.zeros([n_slots, delta, self.n], 'int8'))

        # popu_slots[init_sl_i, 0] = inital


        # TODO 分好槽的popu的f，cost； 索引一个个体的f / c：[slot_i, 个体_i, (0 / 1)]
        f_c_slots = np.array(np.zeros([n_slots, delta, 2], 'float'))
        # f_c_slots[init_sl_i, 0, 0] = self.FS(inital)
        # f_c_slots[init_sl_i, 0, 1] = self.CS(inital)


        ham_slots = np.array(np.zeros([n_slots, delta, 1], 'float'))


        print('..............', wd, slot_wid, L, R)

        t = 0
        T = int(ceil(self.n * self.n * 20))
        print_tn = 2000
        time0 = time.time()

        all_muts = 0
        mutation_dists_sum = 0
        mut_to_slots = np.array(np.zeros(n_slots, 'int'))
        successful_muts = 0
        unsuccessful_muts = 0

        file_name = str(os.path.basename(__file__))
        # with open(file_name + '_result.txt', 'w') as fl:
        #     fl.write('')
        #     fl.flush()
        #     fl.close()

        pm = 1.0 * self.cost.min()

        while t < T:
            t += 2
            if t % print_tn == 0:
                print(t, ' time', time.time() - time0, 's')
                best_f = -np.inf
                best_tupl = 666666666
                for tupl in self.popu_index_tuples:
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

                best_record.append(best_f_c[0])
                t_record.append(t)

                with open(file_name+'_result.txt', 'a') as fl:
                    fl.write(str(t) + '\n')
                    # for i in range(popu_slots.shape[0]):
                    #     for j in range(popu_slots.shape[1]):
                    #         # fl.write(str(popu_slots[i][j])+'\n')
                    #         # pos = self.Position(popu_slots[i][j])
                    #         # for po in pos:
                    #             # fl.write(str(po)+'\t')
                    #         # fl.write('\t\t\t\t\t')
                    #         # fl.write(str(len(pos))+'\t')
                    #         # fl.write(str(ham_slots[i][j])+'\t')
                    #         # fl.write('\n')
                    #     fl.write('\n')

                    # fl.write('\n')
                    # fl.write(str(f_c_slots) + '\n')
                    # fl.write(str(best_tupl) + '\n')
                    fl.write(str(best_f_c) + '\n')
                    # fl.write('\n')
                    fl.flush()
                    fl.close()

                print('f, cost=',  best_f_c, 'Card||=', x_best.sum(), 'popSize=', len(self.popu_index_tuples))
                print('last epoch unsuccessful_mutation rate', int(100*(unsuccessful_muts/all_muts)), '%')
                print('mutation to each slots ratio=', mut_to_slots)
                print('Avg mutation distance=', mutation_dists_sum/all_muts)
                print('----------------------------------------------')

                mut_to_slots = np.array(np.zeros(n_slots, 'int'))
                successful_muts, all_muts, mutation_dists_sum, unsuccessful_muts = 0, 0, 0, 0
                # if t > 5*print_tn:
                #     setMuPT()

                if t % (10*print_tn) == 0:
                    print(f_c_slots)
                    # print(self.global_sum, '------------> global sum')
                    # print(ham_slots)
                #     crit = popu_slots[4]
                #     for p in crit:
                #         print(self.Hamming_Distance(p, crit))

                # if 3*print_tn < t:
                #     plt.plot(t_record, best_record)
                #     plt.show()


            rand_ind = np.random.randint(0, len(self.popu_index_tuples))  # 随机选第几个x，几是相对于popSize而言的
            x_tuple = self.popu_index_tuples[rand_ind]
            x = popu_slots[x_tuple]

            rand_ind = np.random.randint(0, len(self.popu_index_tuples))  # 随机选第几个y，几是相对于popSize而言的
            y_tuple = self.popu_index_tuples[rand_ind]
            y = popu_slots[y_tuple]

            x_ori = np.copy(x)

            x, y = self.cross_over_2_part(x, y)
            # x, y = self.cross_over_partial(x, y)
            # x, y = self.cross_over_uniform(x, y)


            # if t % 2_0000 == 0:
            #     # todo 自适应增大突变率？？？？？？？？？？？？？
            #     if len(best_record) > 2:
            #         if best_record[-1] == best_record[-2]:
            #             print(t, ' kakakakakakakaaaaaaaaaaaaa', pm)
            #             pm *= (1 + 0.05)
                        # pm = pm

            printi = False
            if t % 10000 == 0:
                printi = True

            x = self.mutation_new(x, (B-L+B+R)/2, B-L, B+R, pm, printi)  # x突变
            y = self.mutation_new(y, (B-L+B+R)/2, B-L, B+R, pm, printi)  # y突变


            mutation_dists_sum += np.abs(x - x_ori).sum()

            f_x = float(self.FS(x))  # todo 把hamming distance考量纳入到 cost中？ 还是说f？还是按cowding dist？？
            cost_x = self.CS(x)
            f_y = float(self.FS(y))
            cost_y = self.CS(y)

            # “朝目标选择” Targeted-Selection
            x_slot_index = int(
                np.ceil((cost_x - (B - L)) / slot_wid)  # 向上取整除法
            ) - 1
            y_slot_index = int(
                np.ceil((cost_y - (B - L)) / slot_wid)  # 向上取整除法
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

            self.put_into_popu(x, x_slot_index, f_x, cost_x, popu_slots, f_c_slots, ham_slots, delta)
            self.put_into_popu(y, y_slot_index, f_y, cost_y, popu_slots, f_c_slots, ham_slots, delta)

        # end While
        # 输出答案
        # best_f = -np.inf
        # best_tupl = 666666666
        # for tupl in self.popu_index_tuples:
        #     fc_i = f_c_slots[tupl]
        #     if fc_i[1] > B:
        #         continue
        #     if fc_i[0] > best_f:
        #         best_f = fc_i[0]
        #         best_tupl = tupl
        # x_best = popu_slots[best_tupl]
        # best_f_c = f_c_slots[best_tupl]
        # return x_best, best_f_c


    def popSize(self):
        return len(self.popu_index_tuples)

    def put_into_popu(self, x, x_slot_index, f_x, cost_x, popu_slots, f_c_slots, ham_slots, delta):
        # TODO 把x与slot内所有个体比较 f，（可能需要维护slot全体f,c值的np array）
        worst_x_index = None
        worst_f = np.inf
        worst_cost = -1.0
        x_is_added = False
        for p in range(0, delta):  # p-> (0~5)
            if (popu_slots[x_slot_index, p] == x).all():
                return
            if f_c_slots[x_slot_index, p, 1] == 0:  # (第p个)某旧个体cost==0，即全0gene,说明槽未满
                # 直接把x放进这里
                x_is_added = True
                popu_slots[x_slot_index, p] = x
                f_c_slots[x_slot_index, p, 0] = f_x
                f_c_slots[x_slot_index, p, 1] = cost_x
                self.popu_index_tuples.append((x_slot_index, p))
                self.global_sum += x
                break
            if (f_c_slots[x_slot_index, p, 0] < worst_f
                    or
                    (f_c_slots[x_slot_index, p, 0] == worst_f and f_c_slots[x_slot_index, p, 1] >= worst_cost)
            ):
                # 遍历的第p比当前两方面最差的个体还要 更差（或一样差）
                worst_f = f_c_slots[x_slot_index, p, 0]
                worst_cost = f_c_slots[x_slot_index, p, 1]
                worst_x_index = p

        # if (not x_is_added) and f_x > worst_f:  # x暂未加入，故当前槽已满，但新个体fx > 最差者的f
        #     # x替换最差者
        #     x_is_added = True
        #     self.global_sum = self.global_sum - popu_slots[x_slot_index, worst_x_index] + x
        #     popu_slots[x_slot_index, worst_x_index] = x
        #     f_c_slots[x_slot_index, worst_x_index, 0] = f_x
        #     f_c_slots[x_slot_index, worst_x_index, 1] = cost_x

        if (not x_is_added) and ((f_x == worst_f and cost_x <= worst_cost) or f_x > worst_f):
            only_worst_f = np.inf
            only_worst_pi = []
            for p in range(delta):
                if f_c_slots[x_slot_index, p, 0] < only_worst_f:
                    only_worst_f = f_c_slots[x_slot_index, p, 0]
                    only_worst_pi = [p]
                elif f_c_slots[x_slot_index, p, 0] == only_worst_f:
                    only_worst_pi.append(p)
            if len(only_worst_pi) == 1:
                only_worst_pi = only_worst_pi[0]
                # worst 只有一个
                x_is_added = True
                self.global_sum = self.global_sum - popu_slots[x_slot_index, only_worst_pi] + x
                popu_slots[x_slot_index, only_worst_pi] = x
                f_c_slots[x_slot_index, only_worst_pi, 0] = f_x
                f_c_slots[x_slot_index, only_worst_pi, 1] = cost_x
                return

            # todo 对哪些个体使用Ｈａｍｍｉｎｇ筛选呢？
                # TODO 1. 仅对拥有最差f的个体，忽略cost
                # TODO 2. 仅对拥有最差f与最差cost的个体
                # TODO 3. 对非最优的全部


            # # todo 3.3.3.
            # best_1_f = -1
            # best_1_at = None
            # temp_cost = np.inf
            # for p in range(delta):
            #     if f_c_slots[x_slot_index, p, 0] > best_1_f or (f_c_slots[x_slot_index, p, 0] == best_1_f and f_c_slots[x_slot_index, p, 1] < temp_cost):
            #         best_1_f = f_c_slots[x_slot_index, p, 0]
            #         temp_cost = f_c_slots[x_slot_index, p, 1]
            #         best_1_at = p
            # only_worst_pi = [i for i in range(delta) if i != best_1_at]
            # # todo end 3.3.3.3.3.

            worst_ham = np.inf
            worst_ham_index = None
            temp_global = self.global_sum + x
            temp_popN = self.popSize() + 1
            for pi in only_worst_pi:
                old_xi = popu_slots[x_slot_index, pi]
                ham_i = (np.multiply(temp_popN - temp_global, old_xi) +
                         np.multiply(temp_global, 1 - old_xi)
                        ).sum()
                if ham_i < worst_ham:
                    worst_ham = ham_i
                    worst_ham_index = pi

            x_ham = (np.multiply(temp_popN - temp_global, x) +
                         np.multiply(temp_global, 1 - x)
                        ).sum()
            if x_ham > worst_ham or f_x > worst_f:
                x_is_added = True
                self.global_sum = self.global_sum - popu_slots[x_slot_index, worst_ham_index] + x
                popu_slots[x_slot_index, worst_ham_index] = x
                f_c_slots[x_slot_index, worst_ham_index, 0] = f_x
                f_c_slots[x_slot_index, worst_ham_index, 1] = cost_x


            # # todo 消融实验
            # wowo_c = -1
            # wowo_pi = None
            # for pi in only_worst_pi:
            #     if f_c_slots[x_slot_index, pi, 1] > wowo_c:
            #         wowo_c = f_c_slots[x_slot_index, pi, 1]
            #         wowo_pi = pi
            # popu_slots[x_slot_index, wowo_pi] = x
            # f_c_slots[x_slot_index, wowo_pi, 0] = f_x
            # f_c_slots[x_slot_index, wowo_pi, 1] = cost_x


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
        B=3
        weightMatrix=ReadData(p,filePath)
        nodeNum=np.shape(weightMatrix)[0]
        myObject=ObjectiveIM(weightMatrix,nodeNum)
        B_ = 3
        n_sl = 10

        coo = np.array(myObject.cost)

        myObject.MyPOSS(B_, n_sl, coo.mean(), coo.mean(), delta=10)


