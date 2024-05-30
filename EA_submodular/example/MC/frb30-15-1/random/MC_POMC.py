import sys
import time

import numpy as np
from random import sample
from random import randint, random
from math import pow, log, ceil, fabs, exp


class SUBMINLIN(object):
    def __init__(self, data):
        self.data = data

    def InitDVC(self, n, q):
        self.n = n
        self.cost = [0] * self.n

        ###read cost
        file1=open('cost.txt')
        line=file1.readline()
        items=line.split()
        print
        ########
        for i in range(self.n):
            self.cost[i]=float(items[i])

        '''
        tempElemetn = [i]
        tempElemetn.extend(self.data[i])
        tempValue = len(list(set(tempElemetn))) - q
        if tempValue > 0:
            self.cost[i] = tempValue
        else:
            self.cost[i] = 1
        '''

    def Position(self, s):
        return np.array(np.where(s[0, :] == 1)[1]).reshape(-1)

    def FS(self, s):
        pos = self.Position(s)
        tempSet = []
        for j in pos:
            tempSet.extend(self.data[j])
        tempSet.extend(pos)
        tempSet = list(set(tempSet))
        tempSum = len(tempSet)
        return tempSum

    def CS(self, s):
        pos = self.Position(s)
        tempSum = 0.0
        for item in pos:
            tempSum += self.cost[item]
        return tempSum

    def mutation(self, s):
        s_ori = s
        while 1:
            rand_rate = 1.0 / (self.n)  # the dummy items are considered
            # rand_rate = 0.01
            change = np.random.binomial(1, rand_rate, self.n)
            s = np.abs(s_ori - change)
            if (s != s_ori).any():
                return s

    def POMC(self,B):

        print(self.n)
        print(self.cost)
        cos = np.array(self.cost)
        print('avg cost', cos.sum()/len(self.cost))
        print('sum cost', cos.sum())
        print(B)

        population = np.mat(np.zeros([1, self.n], 'int8'))  # initiate the population
        # population = np.mat(np.random.binomial(1, 0.7, self.n), 'int8')  # initiate the population


        fitness = np.mat(np.zeros([1, 2]))
        popSize = 1
        t = 0  # the current iterate count
        iter = 0
        # T=int(ceil((n+self.constraint)*k*k*exp(1)*exp(1)))
        T = int(ceil(n * n * 40))
        kn = int(self.n * self.n)

        useG = False
        if useG == True:
            print('G!!!')
        else:
            print('F!!!')

        time0 = time.time()
        kn = 20000

        ll = 0.8

        while t < T:
            if iter == kn:
                iter = 0
                resultIndex = -1
                maxValue = float("-inf")
                for p in range(0, popSize):
                    if fitness[p, 1] <= B and fitness[p, 0] > maxValue:
                        maxValue = fitness[p, 0]
                        resultIndex = p
                print(np.ceil(time.time() - time0), 's')
                print(t, 'f c pop', fitness[resultIndex, :],popSize)
                print('| |', population[resultIndex, :].sum(), 'cost', self.CS(population[resultIndex, :]))

                # if t % 80000:
                #     np.set_printoptions(precision=3, suppress=True)
                #     fit = np.array(fitness)
                #     print(fit[np.argsort(fit[:,0])])

                # print(fitness)
            iter += 1
            s = population[randint(1, popSize) - 1, :]  # choose a individual from population randomly
            offSpring = self.mutation(s)  # every bit will be flipped with probability 1/n
            offSpringFit = np.mat(np.zeros([1, 2]))  # comparable value, size, original value
            offSpringFit[0, 1] = self.CS(offSpring)
            if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] > 2.0*B: # or offSpringFit[0, 1] < 0.7*B:
                t += 1
                continue
            offSpringFit[0, 0] = self.FS(offSpring)


            if useG:
                if offSpring.sum() >= 1:
                    # print('?', offSpring.sum(), 'cost', offSpringFit[0,1], 'chuyi', (1.0-(1.0/exp(offSpringFit[0,1]/B))))

                    offSpringFit[0, 0] = offSpringFit[0, 0] / (1.0-(1.0/
                                                                    exp(
                                                                        (abs(offSpringFit[0,1] - B) + ll*B)/B
                                                                    )
                                                                    ))
                else:
                    offSpringFit[0, 0] = 0.00001


            hasBetter = False
            for i in range(0, popSize):
                if (fitness[i, 0] > offSpringFit[0, 0] and fitness[i, 1] <= offSpringFit[0, 1]) or (
                        fitness[i, 0] >= offSpringFit[0, 0] and fitness[i, 1] < offSpringFit[0, 1]):
                    hasBetter = True
                    break
            if hasBetter == False:  # there is no better individual than offSpring
                Q = []
                for j in range(0, popSize):
                    if offSpringFit[0, 0] >= fitness[j, 0] and offSpringFit[0, 1] <= fitness[j, 1]:
                        continue
                    else:
                        Q.append(j)
                # Q.sort()
                fitness = np.vstack((offSpringFit, fitness[Q, :]))  # update fitness
                population = np.vstack((offSpring, population[Q, :]))  # update population
            t = t + 1
            popSize = np.shape(fitness)[0]
        resultIndex = -1
        maxValue = float("-inf")
        for p in range(0, popSize):
            if fitness[p, 1] <= B and fitness[p, 0] > maxValue:
                maxValue = fitness[p, 0]
                resultIndex = p
        return fitness[resultIndex, 0]

    def GS(self,B,offSpringFit):
        if offSpringFit[0,2] >= 1:
            return 1.0*offSpringFit[0,0]/(1.0-(1.0/exp(offSpringFit[0,1]/B)))
        else:
            return 0

    def bin(self,B,size,fitness,popSize,isSmallOrEqual):
        resultList=[]
        if isSmallOrEqual==True:
            for i in range(popSize):
                if fitness[i,2]==size and fitness[i,1]<=B:
                    resultList.append(i)
                    break
        else:
            for i in range(popSize):
                if fitness[i,2]==size and fitness[i,1]>B:
                    resultList.append(i)
                    break
        return resultList


def GetDVCData(fileName):  # node number start from 0
    node_neighbor = []
    i = 0
    file = open(fileName)
    lines = file.readlines()
    while i < 450:
        currentLine = []
        for line in lines:
            items = line.split()
            if int(items[0]) == int(i+1):
                currentLine.append(int(int(items[1])-1))
        node_neighbor.append(currentLine)
        i += 1
    file.close()
    return node_neighbor


if __name__ == "__main__":

    # read data and normalize it
    data = GetDVCData('./../frb30-15-1.mis')
    myObject = SUBMINLIN(data)
    n = 450
    q = 6
    myObject.InitDVC(n, q)  # sampleSize,n,
    B= 7
    myObject.POMC(B)
