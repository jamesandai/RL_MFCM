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
class DI_FSC:
    #初始化参数
    def __init__(self):
        self.fo = FileOperatoruci()
        self.pf = PrintFigures()
        self.data,self.label = self.fo.readIris("../dataset/Iris.data")
        self.c = 3
        self.Max_iter = 100
        self.Eps = 0.00000001
        self.size = len(self.data)
        self.m = 1.5
        self.r = 1.1
        self.alpha = 3.5
        self.beta = 1.3
        self.eu = 0.0001
        self.ew = 0.0001#eu和ew不能像文中设置，要差不多才可以
        self.dim = len(self.data[0])

    #初始化中心矩阵和权重矩阵
    def initialize(self):
        random_V = random.sample(range(self.size),self.c)
        V = self.data[random_V]
        W = [[(1/self.dim)**(1/self.beta) for h in range(self.dim)] for i in range(self.c)]
        U = self.calculate_Membership(V,W)
        return V,W,U

    #计算隶属度矩阵
    def calculate_Membership(self,V,W):
        U = list()
        for i in range(self.size):
            U.append([])
        for k in range(self.size):
            dis_k = list()
            for i in range(self.c):
                dis_ik = 0
                for h in range(self.dim):
                    dis_ik += np.power(W[i][h],self.alpha) * np.power((self.data[k][h] - V[i][h]),2)
                dis_ik = np.sqrt(dis_ik)
                dis_k.append(dis_ik)#点i和所有聚类中心之间的距离列表

            for i in range(self.c):
                temp = list()
                for l in range(self.c):
                    temp.append(np.power((dis_k[i]**2+self.eu) / (dis_k[l]**2+self.eu),self.r / (self.m-self.r)))
                U[k].append((sum(temp)**(-1/self.r)))
        return U

    #计算目标function值J
    def calculate_function(self,U,V,W):
        J = 0
        U_T = list(zip(*U))
        for i in range(self.c):
            for h in range(self.dim):
                J += self.ew * np.power(W[i][h],self.alpha)
        for i in range(self.c):
            for k in range(self.size):
                J += self.ew * np.power(U_T[i][k],self.m)
                dis_ik = 0
                for h in range(self.dim):
                    dis_ik += np.power(W[i][h],self.alpha) * np.power((self.data[k][h] - V[i][h]),2)
                J += np.power(U_T[i][k],self.m) * dis_ik
        return J

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

    #计算权重矩阵W
    def calculate_W(self,U,V):
        U_T = list(zip(*U))
        W = [[] for i in range(self.c)]
        for i in range(self.c):
            h_numerator = []
            for h in range(self.dim):
                temp = 0
                for k in range(self.size):
                    temp += (U_T[i][k] ** self.m) * ((self.data[k][h] - V[i][h]) ** 2)
                h_numerator.append(temp + self.ew)#所有的U（xkh-vih)
            for h in range(self.dim):
                temp = []
                for l in range(self.dim):
                    temp.append((h_numerator[h] / h_numerator[l]) ** (self.beta / (self.alpha-self.beta)))
                W_i_h = sum(temp) ** (-1/self.beta)
                W[i].append(W_i_h)
        return W


    def iteration(self,U,V,W):
        iter = 0
        last_J = np.inf
        while iter <= self.Max_iter:
            iter += 1
            V = self.calculate_Center(U)
            W = self.calculate_W(U,V)
            U = self.calculate_Membership(V, W)
            J = self.calculate_function(U, V, W)
            print(J)
            if np.abs(J - last_J) <= 10**(-6):#若前后两次目标函数的差异不在一个区间的话，继续
                break
            last_J = J
        return U,V,W

    def get_Result(self,U):
        result = list()
        for i in range(self.size):
            max_value,index = max((value,index) for index,value in enumerate(U[i]))
            result.append(index)
        return result



    def run_algorithm(self):
        V,W,U = self.initialize()
        U,V,W = self.iteration(U,V,W)
        result = self.get_Result(U)
        fmi = lambda x, y: fowlkes_mallows_score(x, y)
        nmi = lambda x, y: normalized_mutual_info_score(x, y)
        ari = lambda x, y: adjusted_rand_score(x, y)
        print("%.3f,%.3f,%.3f" % (nmi(self.label, result), ari(self.label, result), fmi(self.label, result)))



if __name__ == "__main__":
    fcm1 = DI_FSC()
    fcm1.run_algorithm()
