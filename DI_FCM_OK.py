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


class DI_FCM:
    # 初始化参数
    def __init__(self):
        self.fo = FileOperatoruci()
        self.pf = PrintFigures()
        self.data, self.label = self.fo.readIris("../dataset/Iris.data")
        self.c = 3
        self.Max_iter = 100
        self.Eps = 0.00001
        self.size = len(self.data)
        self.m = 0.7
        self.r = 0.5
        self.dim = len(self.data[0])

    # 初始化隶属度矩阵 U,n*c
    def initialize(self):
        U = list()
        for i in range(self.size):
            random_list = [random.random() for j in range(self.c)]
            sumamation = np.power(sum(np.power(random_list,self.r)),1/self.r)
            # 满足一个点对所有类的隶属度和为1的条件
            temp_list = [x / sumamation for x in random_list]
            U.append(temp_list)
        return U#这里返回的是n * c的矩阵

    # 计算中心矩阵 V
    def calculate_Center(self, U):
        U_T = list(zip(*U))#c * n的矩阵
        V = list()
        for j in range(self.c):
            x = U_T[j]
            xraised = [np.power(e, self.m) for e in x]  # 对于第k类来说，uik的m次方列表
            V_j_fenmu = sum(xraised)
            V_j_num = list()
            for k in range(self.size):
                uij_yk = [xraised[k] * self.data[k, j] for j in range(self.dim)]
                V_j_num.append(uij_yk)
            V_j_fenzi = map(sum, zip(*V_j_num))#zip(*)是将V_j_num里按列取出构成元组，然后用map函数对zip后的元组做sum操作，也就是在原来V_j_num里的所有列做sum
            # V_j_fenzi = np.sum(V_j_num,axis=0)
            V.append([z / V_j_fenmu for z in V_j_fenzi])
        return V

    def calculate_Membership(self, V):
        U = list()
        for i in range(self.size):
            U.append([])
        for k in range(self.size):
            dis_k = list()
            for i in range(self.c):
                dis_k.append(np.linalg.norm(self.data[k] - V[i]))  # 点i和所有聚类中心之间的距离列表
            for i in range(self.c):
                temp = list()
                for j in range(self.c):
                    temp.append(np.power(dis_k[i] / dis_k[j], 2 * self.r / (self.m - self.r)))
                U[k].append(1 / np.power(np.sum(temp),1/self.r))
        return U

    def iteration(self, U):
        iter = 0
        J_last = np.inf
        while iter <= self.Max_iter:
            iter += 1
            V = self.calculate_Center(U)
            U = self.calculate_Membership(V)
            J = self.calculate_J(U,V)
            print(J)
            if abs(J - J_last) < self.Eps:
                break
            J_last = J
        return U,V

    def calculate_J(self,U,V):
        J = 0
        U_T = list(zip(*U))
        for k in range(self.size):
            for i in range(self.c):
                J += np.power(np.linalg.norm(self.data[k] - V[i]),2) * np.power(U_T[i][k],self.m)
        return J

    def get_Result(self):
        result = list()
        for i in range(self.size):
            max_value, index = max((value, index) for index, value in enumerate(self.U[i]))
            result.append(index)
        return result


    def run_algorithm(self):
        self.U = self.initialize()
        self.U, self.V = self.iteration(self.U)
        self.result = self.get_Result()
        fmi = lambda x, y: fowlkes_mallows_score(x, y)
        nmi = lambda x, y: normalized_mutual_info_score(x, y)
        ari = lambda x, y: adjusted_rand_score(x, y)
        print("%.3f,%.3f,%.3f" % (
        nmi(self.label, self.result), ari(self.label, self.result), fmi(self.label, self.result)))


if __name__ == "__main__":
    fcm1 = DI_FCM()
    fcm1.run_algorithm()
