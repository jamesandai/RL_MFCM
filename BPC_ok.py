import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import fowlkes_mallows_score, normalized_mutual_info_score, adjusted_rand_score
import pandas as pd
from tools.FileOperator import FileOperator
from tools.FileOperatoruci import FileOperatoruci
from tools.PrintFigures import PrintFigures
import sys
import math

sys.path.append("../..")


class BPC:
    #初始化参数
    def __init__(self):
        self.fo = FileOperator()
        self.fouci = FileOperatoruci()
        self.pf = PrintFigures()
        # self.data = self.fo.readButterfly("../dataset/butterfly.csv")
        # self.data,self.label = self.fo.readFourclass("../dataset/fourclass.csv")
        # print(self.data)
        self.K = 30
        # self.data,self.label = self.fouci.readIris("../dataset/iris.data")
        # self.K = 30
        self.data, self.label = self.fouci.readWine("../dataset/wine.data")
        self.pf.printScatter(self.data)
        min_max_scaler = preprocessing.MinMaxScaler()
        self.data = min_max_scaler.fit_transform(self.data)
        self.size,self.dim = self.data.shape
        self.q = 0.9
        self.alpha0 = 1 / self.K

    #运行程序
    def Run(self):
        dis = self.calculate_Distance()
        KNN = self.calculate_Nearest_Neighbors(dis)
        gamma = self.calculate_Gamma(KNN,dis)
        belief = self.calculate_Belief(KNN,gamma,dis)
        delta_distance = self.calculate_Delta_distance(belief,dis)
        # for i in range(self.size):
        #     print(i,belief[i])
        self.pf.printBPEC(belief,delta_distance)
        centers = self.identifyCenters(belief,delta_distance)
        cores = []
        self.neigh = np.zeros(self.size)
        for i in range(self.size):
            if centers[i] != 0:
                cores.append(i)
        result = self.assignDataPoint(dis, belief, centers)
        print(result)
        fmi = lambda x, y: fowlkes_mallows_score(x, y)
        nmi = lambda x, y: normalized_mutual_info_score(x, y)
        ari = lambda x, y: adjusted_rand_score(x, y)
        print("%.3f,%.3f,%.3f" % (
        nmi(self.label, result), ari(self.label, result), fmi(self.label, result)))

    def quantile_exc(self,data,n):
        dic = {}
        a = 1
        data = list(data)
        for i in data:
            dic[a] = i
            a = a + 1
        value = ((a-1)*n)
        return dic[math.ceil(value)]

    #计算距离矩阵
    def calculate_Distance(self):
        dis = np.zeros((self.size,self.size))
        for i in range(self.size):
            for j in range(i+1,self.size):
                dd = np.linalg.norm(self.data[i] - self.data[j])
                dis[i,j] = dd
                dis[j,i] = dd
        return dis

    def calculate_Nearest_Neighbors(self,dis):
        dis_sort = np.argsort(dis,axis=1)
        KNN = dis_sort.tolist()
        for i in range(self.size):
            KNN[i].remove(i)
        KNN = np.array(KNN)
        KNN = KNN[:,:self.K]
        return KNN

    #计算每个点的gamma值
    def calculate_Gamma(self,KNN,dis):
        gamma = np.zeros(self.size)
        for i in range(self.size):
            # gamma[i] = 1 / np.percentile(dis[i,KNN[i]],self.q*100)
            gamma[i] = 1 / self.quantile_exc(dis[i,KNN[i]],self.q)
        return gamma

    #计算每个点成为聚类中心的置信度
    def calculate_Belief(self,KNN,gamma,dis):
        belief = np.zeros(self.size)
        for i in range(self.size):
            belief_i = 1
            for j in KNN[i]:
                belief_i = belief_i * (1 - self.alpha0 * np.exp(-(np.power(gamma[j],2)) * (np.power(dis[i,j],2))))
            belief[i] = 1 - belief_i
        return belief

    #计算delta距离
    def calculate_Delta_distance(self,belief,dis):
        delta_distance = np.zeros(self.size)
        max_belief = np.max(belief)
        for i in range(self.size):
            if belief[i] != max_belief:
                min_dis = np.inf
                for j in range(self.size):
                    if belief[j] > belief[i] and dis[i,j] < min_dis:
                        min_dis = dis[i,j]
                delta_distance[i] = min_dis
        delta_distance[np.where(belief==max_belief)[0]] = np.max(delta_distance)
        return delta_distance

    #识别聚类中心
    def identifyCenters(self,belief,delta_distance):
        thRho = 0.4
        thoDelta = 0.8
        centers = np.zeros(self.size, dtype=np.int)
        cNum = 1
        for i in range(self.size):
            if belief[i] > thRho and delta_distance[i] > thoDelta:#只有大于thDel和thRho的点才能看成是中心
                centers[i] = cNum
                cNum += 1
        return centers

    #分配数据点
    def assignDataPoint(self, dis,belief, result):
        for i in range(self.size):
            dis[i][i] = np.inf
        for i in range(self.size):
            if result[i] == 0:
                result[i] = self.nearestNeighbor(i,dis, belief, result)
            else:
                continue
        return result

    def nearestNeighbor(self, index, dis, belief, result):
        dd = np.inf
        neighbor = -1
        for i in range(self.size):
            if dis[index, i] < dd and belief[index] < belief[i]:
                dd = dis[index, i]
                neighbor = i
        self.neigh[index] = neighbor
        if result[neighbor] == 0:#若没找到，则递归寻找
            result[neighbor] = self.nearestNeighbor(neighbor, dis, belief, result)
        return result[neighbor]

if __name__ == "__main__":
    bpc = BPC()
    bpc.Run()

