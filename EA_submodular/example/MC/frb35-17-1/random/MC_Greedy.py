import sys
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
        #print(self.cost)
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

    def Greedy(self, B):
        self.result = np.mat(np.zeros((1, self.n)), 'int8')
        V_pi=[1]*self.n
        selectedIndex = 0
        while sum(V_pi)>0:
            #print(sum(V_pi))
            f=self.FS(self.result)
            maxVolume = -1
            for j in range(0, self.n):
                if V_pi[j] == 1:
                    self.result[0, j] = 1
                    fv = self.FS(self.result)
                    #cv=self.CS(self.result)
                    tempVolume=1.0*(fv-f)/self.cost[j]
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
            if self.cost[i]<=B:
                tempresult[0,i]=1
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

    def POMC(self,B):
        population = np.mat(np.zeros([1, self.n], 'int8'))  # initiate the population
        fitness = np.mat(np.zeros([1, 2]))
        popSize = 1
        t = 0  # the current iterate count
        iter = 0
        # T=int(ceil((n+self.constraint)*k*k*exp(1)*exp(1)))
        T = int(ceil(n * n * 40))
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
                #print(population[resultIndex,:].sum())
            iter += 1
            s = population[randint(1, popSize) - 1, :]  # choose a individual from population randomly
            offSpring = self.mutation(s)  # every bit will be flipped with probability 1/n
            offSpringFit = np.mat(np.zeros([1, 2]))  # comparable value, size, original value
            offSpringFit[0, 1] = self.CS(offSpring)
            if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] > B:
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
        return fitness[resultIndex, 0]

    def GS(self,B,alpha,offSpringFit):
        if offSpringFit[0,2] >= 1:
            return 1.0*offSpringFit[0,0]/(1.0-(1.0/exp(alpha*offSpringFit[0,1]/B)))
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



    def EAMC(self, B):##just consider cost is less B
        X = np.mat(np.zeros([self.n+1, self.n], 'int8'))  # initiate the population
        Y = np.mat(np.zeros([self.n+1, self.n], 'int8'))  # initiate the population
        Z = np.mat(np.zeros([self.n+1, self.n], 'int8'))  # initiate the population
        W = np.mat(np.zeros([self.n+1, self.n], 'int8'))  # initiate the population
        population =np.mat(np.zeros([1, self.n], 'int8'))
        Xfitness=np.mat(np.zeros([self.n+1, 4]))# f(s), c(s),|s|,g(s)
        Yfitness = np.mat(np.zeros([self.n+1, 4]))  # f(s), c(s),|s|,g(s)
        Zfitness = np.mat(np.zeros([self.n+1, 4]))  # f(s), c(s),|s|,g(s)
        Wfitness = np.mat(np.zeros([self.n+1, 4]))  # f(s), c(s),|s|,g(s)
        Wfitness[:,1]=float("inf")
        offSpringFit = np.mat(np.zeros([1, 4]))  # f(s),c(s),|s|,g(s)
        xysame=[0]*(self.n+1)
        zwsame=[0]*(self.n+1)
        xysame[0]=1
        zwsame[0]=1
        popSize = 1
        t = 0  # the current iterate count
        iter1 = 0
        T = int(ceil(n * n * 20))
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
            s = population[randint(1, popSize) - 1, :]  # choose a individual from population randomly
            offSpring = self.mutation(s)  # every bit will be flipped with probability 1/n
            offSpringFit[0, 1] = self.CS(offSpring)
            offSpringFit[0, 0]=self.FS(offSpring)
            offSpringFit[0, 2] = offSpring[0,:].sum()
            offSpringFit[0, 3]=self.GS(B,1.0,offSpringFit)
            indice=int(offSpringFit[0, 2])
            if offSpringFit[0,2]<1:
                t=t+1
                continue
            isadd1=0
            isadd2=0
            if offSpringFit[0,1]<=B:
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


def GetDVCData(fileName):# node number start from 0
    node_neighbor = []
    i = 0
    file = open(fileName)
    lines = file.readlines()
    while i < 595:
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
    data = GetDVCData('../../frb35-17-1.mis')
    myObject = SUBMINLIN(data)
    n =595
    q = 6
    myObject.InitDVC(n, q)  # sampleSize,n,
    B=1
    print(myObject.Greedy(B))
