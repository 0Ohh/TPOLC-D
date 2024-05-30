import sys
import numpy as np
import datetime
from math import ceil,exp
from random import random,randint

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
        dataFile.close()
        #print('size of node is %d' % (self.n))

    def SetSolution(self, solution):
        self.solution = solution

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
        self.solution = solution
        val = 0
        for i in range(10000):
            val += self.FinalActiveNodes()
        return val / 10000.0

    def Position(self, s):
        return np.array(np.where(s[0, :] == 1)[1]).reshape(-1)

    def FS(self, solution):  # simulate 50 times
        self.solution = solution
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


    def mutation(self, s):
        rand_rate = 1.0 / (self.n)  # the dummy items are considered
        change = np.random.binomial(1, rand_rate, self.n)
        return np.abs(s - change)

    def POMC(self, B):
        population = np.mat(np.zeros([1, self.n], 'int8'))  # initiate the population
        fitness = np.mat(np.zeros([1, 2]))
        popSize = 1
        t = 0  # the current iterate count
        iter = 0
        # T=int(ceil((n+self.constraint)*k*k*exp(1)*exp(1)))
        T = int(ceil(self.n * self.n * 20))
        kn = 2000
        while t < T:
            if iter == kn:
                iter = 0
                resultIndex = -1
                maxValue = float("-inf")
                for p in range(0, popSize):
                    if fitness[p, 1] <= B and fitness[p, 0] > maxValue:
                        maxValue = fitness[p, 0]
                        resultIndex = p
                tempValue=self.EstimateObjective_accurate(population[resultIndex,:])
                print(t, tempValue,fitness[resultIndex,1], popSize)
            iter += 1
            s = population[randint(1, popSize) - 1, :]  # choose a individual from population randomly
            offSpring = self.mutation(s)  # every bit will be flipped with probability 1/n
            offSpringFit = np.mat(np.zeros([1, 2]))  # comparable value, size, original value
            offSpringFit[0, 1] = self.CS(offSpring)
            if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] > 2.0*B:
                t += 1
                continue
            offSpringFit[0, 0] = self.FS(offSpring)
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
        tempValue = self.EstimateObjective_accurate(population[resultIndex, :])
        #print(tempValue, fitness[resultIndex, 1], population[resultIndex, :].sum())
        return tempValue

    def GS(self, B, alpha, offSpringFit):
        if offSpringFit[0, 2] >= 1:
            return 1.0 * offSpringFit[0, 0] / (1.0 - (1.0 / exp(alpha * offSpringFit[0, 1] / B)))
        else:
            return 0



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
        print(myObject.POMC(B))


