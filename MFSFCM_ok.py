import numpy as np
import pandas as pd
import sys
import math
import random
import matplotlib.pyplot as plt
import time

from sklearn import preprocessing
from sklearn.metrics import fowlkes_mallows_score, normalized_mutual_info_score, adjusted_rand_score

sys.path.append("../..")
from tools.FileOperator import FileOperator  # 文件操作类
from tools.FileOperatoruci import FileOperatoruci
from tools.PrintFigures import PrintFigures
class MFSFCM:
    #初始化参数
    def __init__(self):
        self.fo = FileOperatoruci()
        self.pf = PrintFigures()
        # self.data,self.label = self.fo.readWine("../dataset/wine.data")
        self.data,self.label = self.fo.readSegmentation("../dataset/segmentation.data")
        # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        # self.data = min_max_scaler.fit_transform(self.data)
        self.size = len(self.data)
        self.dim = len(self.data[0])
        self.c = 7
        self.Eps = 10**(-6)
        self.m = 2.00

    #初始化隶属度矩阵 U,n*c
    def initialize_U(self):
        U = np.zeros((self.c,self.size))
        for i in range(self.size):
            U[:,i] = [random.random() for j in range(self.c)]
            sumamation = np.sum(U[:,i])
            U[:,i] = U[:,i] / sumamation
        return U

    #计算中心矩阵 V
    def calculate_V(self,U):
        V = np.zeros((self.c,self.dim))
        for i in range(self.c):
            v_i_fenmu = 0
            for j in range(self.size):
                v_i_fenmu += np.power(U[i, j], self.m)
            for s in range(self.dim):
                for j in range(self.size):
                    V[i, s] += np.power(U[i, j], self.m) * self.data[j][s]
                V[i, s] = V[i, s] / v_i_fenmu
        return V

    #计算每个点和各个中心之间的距离
    def calculate_D(self,V):
        D = np.zeros((self.c, self.size))
        for i in range(self.c):
            for j in range(self.size):
                D[i,j] = np.linalg.norm(self.data[j] - V[i])
        return D
    #计算隶属度矩阵
    def calculate_U(self,D):
        U = np.zeros((self.c,self.size))
        for j in range(self.size):
            for i in range(self.c):
                temp = []
                for k in range(self.c):
                    temp.append(np.power(D[i,j] / D[k,j] , 2 / (self.m - 1)))
                U[i,j] = 1 / np.sum(temp)
        return U

    #计算两次V之间的距离
    def calculate_VD(self,V,V_):
        VD = np.zeros(self.c)
        for i in range(self.c):
            VD[i] = np.linalg.norm(V[i] - V_[i])
        return VD

    def calculate_XQt(self,D,VD):
        XQt = []
        sort_D = np.argsort(D,axis=0)
        NC = np.zeros(self.size)#最近的中心
        FC = np.zeros(self.size)#最远的中心
        max_VD = np.max(VD)#聚类中心在两次迭代中变化的最大距离
        for j in range(self.size):
            Dj2 = D[sort_D[1,j],j]#点j和在上一时刻的第二近的中心点之间的距离
            Dj1 = D[sort_D[0,j],j]#点j和在上一时刻最近的中心点之间的距离
            NC[j] = sort_D[0,j]
            FC[j] = sort_D[self.c-1,j]
            VD_j = VD[sort_D[0,j]]
            if Dj2 - max_VD >= Dj1 + VD_j:
                XQt.append(j)
        return XQt,NC,FC

    def calculate_U_(self,U,D,XQt,NC,FC):
        U_ = np.zeros((self.c,self.size))
        for j in range(self.size):
            if j in XQt:
                Ij = int(NC[j])
                U_[Ij,j] = 1/(1+(self.c - 1)*np.power(D[Ij,j] / D[int(FC[j]),j],2/(self.m-1)))
                for i in range(self.c):
                    if i != Ij:
                        U_[i,j] = ((1-U_[Ij,j]) / (1-U[Ij,j]))*U[i,j]
            else:
                for i in range(self.c):
                    U_[i,j] = U[i,j]
        return U_


    def iteration(self,U):
        XQt = [range(1,self.size)]
        dif = np.inf
        V = self.calculate_V(U)
        last_V = V.copy()
        while len(XQt) < self.size and dif >= self.Eps:
            D = self.calculate_D(V)
            U = self.calculate_U(D)
            V_ = self.calculate_V(U)
            VD = self.calculate_VD(V, V_)
            XQt, NC, FC = self.calculate_XQt(D, VD)
            U_ = self.calculate_U_(U,D,XQt, NC, FC)
            V = self.calculate_V(U_)
            dif = 0
            for i in range(self.c):
                dif += np.linalg.norm(V[i] - last_V[i])
            last_V = V.copy()
        D = self.calculate_D(V)
        U = self.calculate_U(D)#最后返回t+1时刻的U
        return U

    def get_Result(self,U):
        result = list()
        for j in range(self.size):
            max_value,index = max((value,index) for index,value in enumerate(U[:,j]))
            result.append(index)
        return result


    def run_algorithm(self):
        U0 = self.initialize_U()
        U = self.iteration(U0)
        result = self.get_Result(U)
        fmi = lambda x, y: fowlkes_mallows_score(x, y)
        nmi = lambda x, y: normalized_mutual_info_score(x, y)
        ari = lambda x, y: adjusted_rand_score(x, y)
        print("%.3f,%.3f,%.3f" % (nmi(self.label, result), ari(self.label, result), fmi(self.label, result)))



if __name__ == "__main__":
    mfsfcm = MFSFCM()
    mfsfcm.run_algorithm()
