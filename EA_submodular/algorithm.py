import sys
import time

import numpy as np
from random import sample
from random import randint, random
from math import pow, log, ceil, fabs, exp


class SUBMINLIN(object):
    def __init__(self):

    def Position(self, s):
        return np.array(np.where(s[0, :] == 1)[1]).reshape(-1)

    def FS(self, s):
        return 1

    def CS(self, s):
        return 1

    def Greedy(self, B):
        self.result = np.mat(np.zeros((1, self.n)), 'int8')
        V_pi=[1]*self.n
        selectedIndex = 0
        while sum(V_pi)>0:
            f=self.FS(self.result)
            c=self.CS(self.result)
            maxVolume = -1
            for j in range(0, self.n):
                if V_pi[j] == 1:
                    self.result[0, j] = 1
                    fv = self.FS(self.result)
                    cv=self.CS(self.result)
                    #cv=self.CS(self.result)
                    tempVolume=1.0*(fv-f)/(cv-c)
                    if tempVolume > maxVolume:
                        maxVolume = tempVolume
                        selectedIndex = j
                    self.result[0, j] = 0
            self.result[0,selectedIndex]=1
            if self.CS(self.result)>B:
                self.result[0, selectedIndex] = 0
            V_pi[selectedIndex]=0

        tempMax=0.
        tempresult=np.mat(np.zeros((1, self.n)), 'int8')
        for i in range(self.n):
            tempresult[0,i]=1
            if self.CS(tempresult)<=B:
                tempVolume=self.FS(tempresult)
                if tempVolume>tempMax:
                    tempMax=tempVolume
            tempresult[0, i] = 0
        tempmax1=self.FS(self.result)
        if tempmax1>tempMax:
            return tempmax1
        else:
            return tempMax


    def mutation(self, s):
        rand_rate = 1.0 / (self.n)  # the dummy items are considered
        change = np.random.binomial(1, rand_rate, self.n)
        return np.abs(s - change)


    def mutation_new(self, s, Tar, l_bound, r_bound, der=-1):
        # 保证期望值
        x_ori = np.copy(s)
        nn = int(s.shape[0])
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

            if l_bound < self.CS(x) < r_bound and (x != x_ori).any():
                # if r_bound and (s != s_ori).any():
                # if 1:
                return x


    def MyPOSS(self, B, n_slots, L, R=None, delta=5):
        print(self.n)
        print(self.cost)
        print(self.B)
        if R is None: R = L

        popu_index_tuples = [(0, 0)]  # TODO 一个list[tuples]，整个popu的每个个体对应popu_slots中的index(slot_i, 个体_i)
                                # TODO 这个列表只会append，不会删东西
        # popSize = 1
        # TODO 分好槽的popu，索引一个个体：[slot_i, 个体_i, :]
        popu_slots = np.array(np.zeros([n_slots, delta, self.n], 'int8'))
        # TODO 分好槽的popu的f，cost； 索引一个个体的f / c：[slot_i, 个体_i, (0 / 1)]
        f_c_slots = np.array(np.zeros([n_slots, delta, 2], 'int8'))
        slot_wid = (L + R) / n_slots

        t = 0
        T = int(ceil(self.n * 100))
        print_tn = 10000
        time0 = time.time()
        while t < T:
            t += 1
            if t % print_tn == 0:
                print(t, ' time', time.time() - time0, 's')
                best_f = -np.inf
                best_tupl_index = 666666666
                for tupl in popu_index_tuples:
                    fc_i = f_c_slots[tupl]
                    if fc_i[1] > B:
                        continue
                    if fc_i[0] > best_f:
                        best_f = fc_i[0]
                        best_tupl_index = tupl
                x_best = popu_slots[
                    popu_index_tuples[best_tupl_index]
                ]
                best_f_c = f_c_slots[
                    popu_index_tuples[best_tupl_index]
                ]
                print('f, cost ',  best_f_c)

            rand_ind = randint(1, len(popu_index_tuples) - 1) # 随机选第几个，几是相对于popSize而言的
            x_tuple = popu_index_tuples[rand_ind]
            x = popu_slots[x_tuple]
            x = self.mutation_new(x, B)  # x突变

            f_x = self.FS(x)
            cost_x = self.CS(x)

            # “朝目标选择” Targeted-Selection
            x_slot_index = int(
                (cost_x - (B - L)) // slot_wid  # 向下取整除法
            )

            # slot_for_x = popu_slots[cost_x_slot_index]
            # TODO 把x与slot内所有个体比较 f，（可能需要维护slot全体f,c值的np array）
            worst_x_index = None
            worst_f = np.inf
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
                if f_c_slots[x_slot_index, p, 0] < worst_f:
                    worst_x_index = p
            if (not x_is_added) and f_x > worst_f: # x暂未加入，故当前槽已满，但新个体fx > 最差者的f
                # x替换最差者
                x_is_added = True
                popu_slots[x_slot_index, worst_x_index] = x
                f_c_slots[x_slot_index, worst_x_index, 0] = f_x
                f_c_slots[x_slot_index, worst_x_index, 1] = cost_x
                popu_index_tuples.append((x_slot_index, worst_x_index))
        # end While
        # 输出答案
        best_f = -np.inf
        best_tupl_index = 666666666
        for tupl in popu_index_tuples:
            fc_i = f_c_slots[tupl]
            if fc_i[1] > B:
                continue
            if fc_i[0] > best_f:
                best_f = fc_i[0]
                best_tupl_index = tupl
        x_best = popu_slots[
            popu_index_tuples[best_tupl_index]
        ]
        best_f_c = f_c_slots[
            popu_index_tuples[best_tupl_index]
        ]
        return x_best, best_f_c



    def POMC(self,B):
        # 初始化popu，1行n列，即1个个体；后面增多变成p行
        population = np.mat(np.zeros([1, self.n], 'int8'))  # initiate the population
        # fitness，1行2列；后面增多变成p行
        fitness = np.mat(np.zeros([1, 2]))
        popSize = 1
        t = 0  # the current iterate count
        # iter未知作用
        iter = 0
        # T=int(ceil((n+self.constraint)*k*k*exp(1)*exp(1)))
        # T=循环数=10n^2
        T = int(ceil(self.n *self.n * 10))
        # iter每到kn=n^2就干一件事（可能是打印目前最好），然后iter=0
        kn = int(self.n * self.n)
        while t < T:
            if iter == kn:
                iter = 0
                resultIndex = -1
                maxValue = float("-inf")
                for p in range(0, popSize):
                    if fitness[p, 1] <= B and fitness[p, 0] > maxValue:
                        maxValue = fitness[p, 0]
                        resultIndex = p
                print(fitness[resultIndex, :],population[resultIndex,:].sum())
            iter += 1

            # 随机从popu中选个体s（np.mat，1行n列）
            s = population[randint(1, popSize) - 1, :]
            offSpring = self.mutation(s)  # every bit will be flipped with probability 1/n

            # offSpringFit放个体的[cost, f]
            offSpringFit = np.mat(np.zeros([1, 2]))  # comparable value, size, original value
            offSpringFit[0, 1] = self.CS(offSpring)
            if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] >= 2*B: # cost=0或cost>=2B就舍弃，去进行下一iter
                t += 1
                continue
            offSpringFit[0, 0] = self.FS(offSpring)
            hasBetter = False
            for i in range(0, popSize):  # Loop整个popu，找是否有比当前子代个体更好的旧人
                    # f                                      cost
                if (fitness[i, 0] > offSpringFit[0, 0] and fitness[i, 1] <= offSpringFit[0, 1]) or (
                    fitness[i, 0] >= offSpringFit[0, 0] and fitness[i, 1] < offSpringFit[0, 1]):
                    hasBetter = True
                    break
            if hasBetter == False:  # there is no better individual than offSpring
                Q = []
                for j in range(0, popSize):
                    #  f现 >= f旧j                           且  cost现 <= cost旧j
                    if offSpringFit[0, 0] >= fitness[j, 0] and offSpringFit[0, 1] <= fitness[j, 1]:
                        continue
                    else:
                        Q.append(j)   # Q记录所有当前新人 不能支配 的旧人index们；Q为存活者编号
                # Q.sort()
                fitness = np.vstack((offSpringFit, fitness[Q, :]))  # update fitness
                population = np.vstack((offSpring, population[Q, :]))  # update population
            t += 1
            popSize = np.shape(fitness)[0]
        # While结束
        # 找到最好个体（输出答案）
        resultIndex = -1
        maxValue = float("-inf")
        for p in range(0, popSize):
            if fitness[p, 1] <= B and fitness[p, 0] > maxValue:
                maxValue = fitness[p, 0]
                resultIndex = p
        return fitness[resultIndex, 0]

    def GS(self,B,alpha,offSpringFit):
        # 那个新g，= f/(1-e^(c/B))
        if offSpringFit[0,2] >= 1:
            return 1.0*offSpringFit[0,0]/(1.0-(1.0/exp(alpha*offSpringFit[0,1]/B)))
        else:
            return 0

    def EAMC(self, B):  ##just consider cost is less B  （我注：是说“只考虑cost<B”的意思吗）
        X = np.mat(np.zeros([self.n+1, self.n], 'int8'))  # initiate the population
        Y = np.mat(np.zeros([self.n+1, self.n], 'int8'))  # initiate the population
        # TODO popu Z, W这里没用，有的问题用了，不知何物
        Z = np.mat(np.zeros([self.n+1, self.n], 'int8'))  # initiate the population
        W = np.mat(np.zeros([self.n+1, self.n], 'int8'))  # initiate the population
        population =np.mat(np.zeros([1, self.n], 'int8'))
        Xfitness = np.mat(np.zeros([self.n+1, 4]))# f(s), c(s),|s|,g(s)
        Yfitness = np.mat(np.zeros([self.n+1, 4]))  # f(s), c(s),|s|,g(s)
        Zfitness = np.mat(np.zeros([self.n+1, 4]))  # f(s), c(s),|s|,g(s)
        Wfitness = np.mat(np.zeros([self.n+1, 4]))  # f(s), c(s),|s|,g(s)
        Wfitness[:,1]=float("inf")
        offSpringFit = np.mat(np.zeros([1, 4]))  # f(s),c(s),|s|,g(s)

        # TODO 未明
        xysame=[0]*(self.n+1)
        zwsame=[0]*(self.n+1)
        xysame[0]=1
        zwsame[0]=1
        popSize = 1
        t = 0  # the current iterate count
        iter1 = 0
        T = int(ceil(self.n *self.n * 10))
        kn = int(self.n*self.n)
        while t < T:
            if iter1 == kn:
                iter1 = 0
                resultIndex = -1
                maxValue = float("-inf")
                for p in range(0, self.n+1):
                    if Yfitness[p, 1] <= B and Yfitness[p, 0] > maxValue:
                        maxValue = Yfitness[p, 0]
                        resultIndex = p
                print(Yfitness[resultIndex, :],popSize)
            iter1 += 1

            # 从popu随机挑一个，然后突变
            s = population[randint(1, popSize) - 1, :]
            offSpring = self.mutation(s)
            # 计算f，cost，|s|，最后g
            offSpringFit[0, 0]=self.FS(offSpring)
            offSpringFit[0, 1] = self.CS(offSpring)
            offSpringFit[0, 2] = offSpring[0,:].sum()
            offSpringFit[0, 3]=self.GS(B,1.0,offSpringFit)
            indice=int(offSpringFit[0, 2]) # s中1的个数
            if offSpringFit[0,2]<1:  # 空集 跳过
                t=t+1
                continue
            isadd1=0
            isadd2=0
            if offSpringFit[0,1]<=B:  # cost小于B，则：
                if offSpringFit[0, 3]>=Xfitness[indice,3]:
                    X[indice,:]=offSpring
                    Xfitness[indice,:]=offSpringFit
                    isadd1=1
                if offSpringFit[0, 0]>=Yfitness[indice,0]:
                    Y[indice,:]=offSpring
                    Yfitness[indice, :] = offSpringFit
                    isadd2=1
                if isadd1+isadd2==2:
                    xysame[indice] = 1
                else:
                    if isadd1+isadd2==1:
                        xysame[indice] = 0
            # count the population size
            tempSize=1 #0^n is always in population
            for i in range(1,self.n+1):
                if Xfitness[i,2]>0:
                    if Yfitness[i,2]>0 and xysame[i]==1:#np.linalg.norm(X[i,:]-Y[i,:])==0: #same
                        tempSize=tempSize+1
                    if Yfitness[i,2]>0 and xysame[i]==0:#np.linalg.norm(X[i,:]-Y[i,:])>0:
                        tempSize=tempSize+2
                    if Yfitness[i,2]==0:
                        tempSize=tempSize+1
                else:
                    if Yfitness[i,2]>0:
                        tempSize=tempSize+1
            if popSize!=tempSize:
                population=np.mat(np.zeros([tempSize, self.n], 'int8'))
            popSize=tempSize
            j=1
            # merge the X,Y,Z,W
            for i in range(1,self.n+1):
                if Xfitness[i, 2] > 0:
                    if Yfitness[i, 2] > 0 and xysame[i] == 1:
                    #if Yfitness[i, 2] > 0 and np.linalg.norm(X[i, :] - Y[i, :]) == 0:  # same
                        population[j,:]=X[i,:]
                        j=j+1
                    if Yfitness[i, 2] > 0 and xysame[i] == 0:
                    #if Yfitness[i, 2] > 0 and np.linalg.norm(X[i, :] - Y[i, :]) > 0:
                        population[j, :] = X[i, :]
                        j=j+1
                        population[j, :] = Y[i, :]
                        j=j+1
                    if Yfitness[i, 2] == 0:
                        population[j, :] = X[i, :]
                        j = j + 1
                else:
                    if Yfitness[i, 2] > 0:
                        population[j, :] = Y[i, :]
                        j = j + 1
            t=t+1
        resultIndex = -1
        maxValue = float("-inf")
        for p in range(0, self.n+1):
            if Yfitness[p, 1] <= B and Yfitness[p, 0] > maxValue:
                maxValue = Yfitness[p, 0]
                resultIndex = p
        print(Yfitness[resultIndex, :],popSize)


