import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import fowlkes_mallows_score, normalized_mutual_info_score, adjusted_rand_score, rand_score
import pandas as pd
import time
from sklearn import preprocessing
from sklearn.metrics import fowlkes_mallows_score, normalized_mutual_info_score, adjusted_rand_score
import sys
import math
sys.path.append("../..")
from tools.FileOperator import FileOperator  # 文件操作类
from tools.FileOperatoruci import FileOperatoruci
from tools.PrintFigures import PrintFigures

class RL_FCM:
    def __init__(self):
        self.Eps = 10**(-3)
        self.pf = PrintFigures()
        self.fouci = FileOperatoruci()
        self.fo = FileOperator()
        # self.data, self.label = self.fo.readDatawithLabel2("../datasets/R15.txt")#15个,(100,10,0.986,0.997,0.980,0.978,第13次迭代)15
        # self.name = "R15"
        # self.data,self.label = self.fo.readDatawithLabel("../datasets/data/Synthetic/twenty.arff")#(100,10,1.000,1.000,1.000,第15次)20
        # self.name = "twenty"
        # self.data, self.label = self.fo.readDatawithLabel2("../datasets/flame.txt")#(100,100,0.427,0.732,0.742,0.465,第182次)2
        # self.name = "flame"
        # self.data, self.label = self.fo.readDatawithLabel("../datasets/fourty.arff")#(100，100，1.000,1.000,1.000,1.000，第22次)40
        # self.name = "fourty"
        # self.data, self.label = self.fo.readDatawithLabel("../datasets/data/Synthetic/elliptical_10_2.arff")#(100，100，0.915,0.970,0.860,0.841，第111次)9
        # self.name = "elliptical"
        # self.data, self.label = self.fo.readData3D("../datasets/data/Synthetic/hepta.arff")#(100,1000,1.000,1.000,1.000,1.000,13次)10
        # self.name = "hepta"
        # self.data, self.label = self.fo.readDatawithLabel("../datasets/data/Synthetic/sizes3.arff")#8(100,100,0.623,0.642,0.507,0.264,206)
        # self.name = "size3"
        #
        # self.data, self.label = self.fo.readDatawithLabel("../datasets/data/Synthetic/sizes4.arff")#9(100,100,0.550,0.562,0.456,0.190,190)
        # self.name = "size4"
        # self.data, self.label = self.fo.readDatawithLabel("../datasets/data/Synthetic/sizes5.arff")#8(100,100,0.534,0.526,0.471,0.181,137)
        # self.name = "size5"
        # self.data, self.label = self.fo.readDatawithLabel(
        #     "../datasets/data/Synthetic/triangle1.arff")  #

        # self.pf.printScatter(self.data)
        # self.pf.print_Dcores_order(self.data)
        # self.data,self.label = self.fouci.readIris("../uci datasets/iris.data")#3,(100,1000,0.778,0.906,0.856,0.786,17)
        self.data,self.label = self.fouci.readSeed("../uci datasets/seeds_dataset.txt")#3(100,100,0.867,0.703,0.700,0.800,51)
        # self.data, self.label = self.fouci.readDivorce("../uci datasets/divorce.csv")#2(100,100,0.862,0.954,0.954,0.908,26)
        # self.data,self.label = self.fouci.readLymphography("../uci datasets/lymphography.data")#2(100,100,0.056,0.531,0.532,0.065,18)
        # self.data,self.label = self.fouci.readParkinsons("../uci datasets/parkinsons.data")#2个(无法识别)
        # self.data, self.label = self.fouci.readSegmentation(
        #     "../uci datasets/segmentation.data")  # 7个(100,100,0.412,0.858,0.037,0.002,176)
        # self.data , self.label = self.fouci.readWine("../uci datasets/wine.data")#3个(无法识别)
        # self.data, self.label = self.fouci.readColoumn("../uci datasets/coloumn")#3个（无法识别）
        # self.data, self.label = self.fouci.readNewthyroid("../uci datasets/new-thyroid.data")#3个（无法处理）
        # self.data, self.label = self.fouci.readWdbc("../uci datasets/wdbc.data")#2个（无法处理）

        self.size,self.dim = self.data.shape
        self.c = self.size#初始聚类数为全部

    def Run(self):
        U,V = self.Iteration()
        result = np.argmax(U,axis=1)
        fmi = lambda x, y: fowlkes_mallows_score(x, y)
        nmi = lambda x, y: normalized_mutual_info_score(x, y)
        ri = lambda x, y: rand_score(x, y)
        ari = lambda x, y: adjusted_rand_score(x, y)
        print("%.3f,%.3f,%.3f,%.3f" % (ri(self.label, result), nmi(self.label, result), ari(self.label, result),fmi(self.label,result)))

    def Iteration(self):
        A, r1, r2, r3, V_last = self.Initial()
        c_last = self.size
        t = 1
        class_num = np.zeros(100)
        flag = 0
        A_nor = A.copy()
        while True:
            print(c_last,t)
            U = self.calculate_U(A_nor, V_last, r1, r2,c_last)
            r1 = np.exp(-(t)/1000)  # 更新r1
            r2 = np.exp(-(t) / 1000)  # 更新r2(iris)
            A = self.calculate_A(U, A_nor, r3, r1,c_last)  # 更新概率矩阵
            if flag == 0:
                r3 = self.calculate_r3(A, A_nor,U,c_last)  # 更新r3
            A_nor, U_nor,c = self.select_class(A, U,c_last)  # 删除了不应该存在的类后的概率矩阵和隶属度矩阵
            if flag == 0:
                if t >= 100:
                    if class_num[t%100] == c:
                        r3 = 0
                        flag = 1
                class_num[(t-1) % 100] = c
            V = self.calculate_V(U_nor,c)
            max_dist = 0
            for i in range(c):
                if np.linalg.norm(V[i] - V_last[i]) > max_dist:
                    max_dist = np.linalg.norm(V[i] - V_last[i])
            if max_dist < self.Eps:
                break
            V_last = V.copy()
            t += 1
            c_last = c
            # if c <= 40:
            #     self.pf.print_fcm_center(self.data, V, t, self.name)
            #     result = np.argmax(U, axis=1)
            #     self.pf.printScatter_Color(self.data, result, t, self.name)
        print(c)
        return U,V

    def Initial(self):#初始化中心矩阵，概率矩阵等
        A = [1 / self.size for i in range(self.size)]
        r1 = 1
        r2 = 1
        r3 = 1
        V = []
        for i in range(self.size):
            current_center = []
            for s in range(self.dim):
                current_center.append(self.data[i][s])
            V.append(current_center)
        return A,r1,r2,r3,V


    def calculate_D(self,V,c):
        D = np.zeros((self.size,c))
        for i in range(self.size):
            for j in range(c):
                D[i,j] = np.linalg.norm(self.data[i] - V[j])
        return D


    def calculate_U(self,A,V,r1,r2,c):
        U = []
        D = self.calculate_D(V,c)
        for i in range(self.size):
            current = []
            U_fenmu = 0
            for t in range(c):
                U_fenmu += np.exp((-np.power(D[i,t],2) + r1 * math.log(A[t])) / r2)
            for k in range(c):
                U_fenzi = np.exp((-np.power(D[i,k],2) + r1 * math.log(A[k])) / r2)
                current.append(U_fenzi / U_fenmu)
            U.append(current)
        return U

    def calculate_A(self,U,A_,r3,r1,c):
        A = []
        sum_ = 0
        for t in range(c):
            sum_ += A_[t] * math.log(A_[t])
        for k in range(c):
            sum1 = 0
            for j in range(self.size):
                sum1 += U[j][k]
            one_part = (1 / self.size) * sum1
            two_part = (r3 / r1) * A_[k] * (math.log(A_[k]) - sum_)
            A.append(one_part + two_part)
        return A

    # 存疑
    def calculate_r3(self,A,A_,U,c):
        one_part = 0
        sum_two = 0
        mix_pro = min(1,2 / np.power(self.dim,np.round(self.dim / 2-1)))
        for k in range(c):
            one_part += np.exp(-self.size * mix_pro * abs(A[k] - A_[k]))
        one_part = one_part / c
        for t in range(c):
            sum_two += A_[t] * math.log(A_[t])
        sum_two = sum_two * (-np.max(A_))
        max_d = 0
        for k in range(c):
            sum1 = 0
            for j in range(self.size):
                sum1 += U[j][k]
            sum_U = (1 / self.size) * sum1
            if sum_U > max_d:
                max_d = sum_U
        two_part = (1 - sum_U) / sum_two
        r3 = min(one_part,two_part)
        return r3

    def select_class(self,A,U,c):
        discard_class = []
        for i in range(c):
            if A[i] < 1 / self.size:
                discard_class.append(i)
        discard_length = len(discard_class)
        sum_A = 0
        A_nor = []
        U_nor = []
        for i in range(c):
            if i not in discard_class:
                sum_A += A[i]
        for i in range(c):
            if i not in discard_class:
                A_nor.append(A[i] / sum_A)
        for i in range(self.size):
            sum_Ui = 0
            for k in range(c):
                if k not in discard_class:
                    sum_Ui += U[i][k]
            current = []
            for k in range(c):
                if k not in discard_class:
                    current.append(U[i][k] / sum_Ui)
            U_nor.append(current)
        c = c - discard_length
        return A_nor,U_nor,c

    def calculate_V(self,U,c):
        V = np.zeros((c,self.dim))
        for k in range(c):
            V_fenmu = 0
            for i in range(self.size):
                V_fenmu += U[i][k]
            for s in range(self.dim):
                for i in range(self.size):
                    V[k][s] += U[i][k] * self.data[i][s]
                V[k][s] = V[k][s] / V_fenmu
        return V








if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    rlfcm = RL_FCM()
    rlfcm.Run()