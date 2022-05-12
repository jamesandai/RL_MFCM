import numpy as np
import pandas as pd
import sys
import math
import random
import matplotlib.pyplot as plt
import time

from sklearn.metrics import fowlkes_mallows_score, normalized_mutual_info_score, adjusted_rand_score

sys.path.append("../..")
from tools.FileOperator import FileOperator  # 文件操作类
from tools.FileOperatoruci import FileOperatoruci
from tools.PrintFigures import PrintFigures
class fcm:
    #初始化参数
    def __init__(self):
        self.fo = FileOperatoruci()
        self.pf = PrintFigures()
        self.data,self.label = self.fo.readIris("../dataset/Iris.data")
        self.c = 3
        self.Max_iter = 100
        self.Eps = 0.00000001
        self.size = len(self.data)
        self.m = 2.00
        self.dim = len(self.data[0])

    #初始化隶属度矩阵 U,n*c
    def initialize(self):
        U = list()
        for i in range(self.size):
            random_list = [random.random() for j in range(self.c)]
            sumamation = sum(random_list)
            #满足一个点对所有类的隶属度和为1的条件
            temp_list = [x / sumamation for x in random_list]
            U.append(temp_list)
        return U

    #计算中心矩阵 V
    def calculate_Center(self,U):
        U_T = list(zip(*U))
        V = list()
        for j in range(self.c):
            x = U_T[j]
            xraised = [np.power(e,self.m) for e in x]#对于第k类来说，uik的m次方列表
            V_j_fenmu = sum(xraised)
            V_j_num = list()
            for k in range(self.size):
                uij_yk = [xraised[k] * self.data[k,j] for j in range(self.dim)]
                V_j_num.append(uij_yk)
            V_j_fenzi = map(sum, zip(*V_j_num))
            V.append([z / V_j_fenmu for z in V_j_fenzi])
        return V

    def calculate_Membership(self,V):
        U = list()
        for i in range(self.size):
            U.append([])
        for k in range(self.size):
            dis_k = list()
            for i in range(self.c):
                dis_k.append(np.linalg.norm(self.data[k] - V[i]))#点i和所有聚类中心之间的距离列表
            for i in range(self.c):
                temp = list()
                for j in range(self.c):
                    temp.append(np.power(dis_k[i] / dis_k[j],2/(self.m-1)))
                U[k].append(1/sum(temp))
        return U

    def iteration(self,U):
        iter = 0
        while iter <= self.Max_iter:
            iter += 1
            V = self.calculate_Center(U)
            U_last = U.copy()
            U = self.calculate_Membership(V)
            juli = 0
            for k in range(self.size):
                for i in range(self.c):
                    juli += np.power((U[k][i] - U_last[k][i]),2)
            if (np.sqrt(juli)) <= self.Eps:
                return U,V

    def get_Result(self,U):
        result = list()
        for i in range(self.size):
            max_value,index = max((value,index) for index,value in enumerate(U[i]))
            result.append(index)
        return result

    #-------------------计算CFE index------------------------
    def calculate_FE(self,U):
        FE = 0
        U_T = list(zip(*U))
        for i in range(self.c):
            for j in range(self.size):
                FE += U_T[i][j]*np.log(U_T[i][j]) + (1-U_T[i][j]) * np.log(1-U_T[i][j])
        FE = -1/(self.c*self.size*np.log(2)) * FE
        return FE

    #计算两两类之间的所有点的隶属度的差异，越大越好
    def calculate_SFCE(self,U):
        SFCE = 0
        U_T = list(zip(*U))
        for i in range(self.c):
            for j in range(i,self.c):
                for n in range(self.size):
                    SFCE += (U_T[i][n] * np.log(U_T[i][n] / (0.5 * U_T[i][n] + 0.5 * U_T[j][n])) + (1 - U_T[i][n]) * np.log((1 - U_T[i][n]) / (1 - 0.5 * U_T[i][n] - 0.5 * U_T[j][n]))) + \
                            (U_T[j][n] * np.log(U_T[j][n] / (0.5 * U_T[j][n] + 0.5 * U_T[i][n])) + (1 - U_T[j][n]) * np.log((1 - U_T[j][n]) / (1 - 0.5 * U_T[j][n] - 0.5 * U_T[i][n])))
        SFCE = 2/(self.c*(self.c-1)) * SFCE
        return SFCE

    #计算每个类中的点和中心之间的距离
    def calculate_WGSD(self,result,V):
        label = []
        WSGD = 0
        for i in range(self.c):
            label.append([])
        for i in range(self.size):
            label[int(result[i])].append(i)
        for i in range(self.c):
            for j in label[i]:
                WSGD += np.linalg.norm(self.data[j] - V[i]) ** 2
        return WSGD

    #计算所有类的中心和全局中心之间的距离
    def calculate_BGSD(self,result,V):
        label = []
        for i in range(self.c):
            label.append([])
        for i in range(self.size):
            label[int(result[i])].append(i)
        V_tot = np.mean(self.data,axis=0)
        BGSD = 0
        for i in range(self.c):
            BGSD += len(label[i]) * np.linalg.norm(V[i] - V_tot) ** 2 / self.size
        print(BGSD)
        return BGSD

    #计算CFE值
    def calculate_CFE(self,FE,SFCE,WGSD,BGSD):
        CH = BGSD / (self.c - 1) * (self.size - self.c) / WGSD
        MC = SFCE / FE
        CFE = 0.5 * (MC + CH)
        return CFE

    #-----------------------计算SMI index---------------------------------

    #计算所有类中，类内模糊隶属度加权距离和最大
    def calculate_Co(self,V,U):
        U = np.array(U)
        max_Co = 0
        for k in range(self.c):
            sum_k = 0
            for i in range(self.size):
                sum_k += U[i][k] ** 2 * np.linalg.norm(self.data[i] - V[k]) ** 2
            sum_k = sum_k / np.sum(U[:,k])
            if max_Co < sum_k:
                max_Co = sum_k
        Co = (self.c - 1) * max_Co
        return Co

    def calculate_S(self,result):
        label = []
        min_S = np.inf
        for i in range(self.c):
            label.append([])
        for i in range(self.size):
            label[int(result[i])].append(i)
        for s in range(self.c-1):
            for t in range(s+1,self.c):
                dis_Cs_Ct = np.inf
                for i in label[s]:
                    for j in label[t]:
                        if dis_Cs_Ct > np.linalg.norm(self.data[i] - self.data[j]) ** 2:
                            dis_Cs_Ct = np.linalg.norm(self.data[i] - self.data[j]) ** 2
                if min_S > dis_Cs_Ct:
                    min_S = dis_Cs_Ct
        print(min_S)
        return min_S



    def run_algorithm(self):
        U = self.initialize()
        U,V = self.iteration(U)
        result = self.get_Result(U)
        # FE = self.calculate_FE(U)
        # SFCE = self.calculate_SFCE(U)
        # WGSD = self.calculate_WGSD(result,V)
        # BGSD = self.calculate_BGSD(result,V)
        # CFE = self.calculate_CFE(FE,SFCE,WGSD,BGSD)

        Co = self.calculate_Co(V,U)
        S = self.calculate_S(result)
        SMI = Co / S
        print(SMI)

        fmi = lambda x, y: fowlkes_mallows_score(x, y)
        nmi = lambda x, y: normalized_mutual_info_score(x, y)
        ari = lambda x, y: adjusted_rand_score(x, y)
        print("%.3f,%.3f,%.3f" % (nmi(self.label, result), ari(self.label, result), fmi(self.label, result)))



if __name__ == "__main__":
    fcm1 = fcm()
    fcm1.run_algorithm()
