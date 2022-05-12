import math
import sys

from sklearn import preprocessing

import numpy as np
from tools.PrintFigures import PrintFigures  # 绘图操作类
from tools.FileOperator import FileOperator  # 文件操作类
from tools.FileOperatoruci import FileOperatoruci
from sklearn.metrics import normalized_mutual_info_score, fowlkes_mallows_score, adjusted_rand_score
import time
sys.path.append("../..")

class IO_DPC:
    MAX = 1000000
    fo = FileOperatoruci()
    pf = PrintFigures()

    # 1 main function of CFSFDP
    neigh = None

    def __init__(self):
        self.data, self.label = self.fo.readIris("../../dataset/Iris.data")
        self.size,self.dim = self.data.shape
        self.neigh = np.zeros(self.size)
        self.dc = 0

    def runAlgorithm(self):
        # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        # points = min_max_scaler.fit_transform(points)
        fcl = []
        dis, dist = self.Get_distance(self.data)
        percent = 2
        position = int(len(dist) * percent / 100)  # Number of neighbors
        sortedll = np.sort(dist)  #
        self.dc = sortedll[position]  # Get the minimum distance of the neighbor as the cutoff distance（找到截止距离，计算方法就是在距离列表里n*(n-1)/2*0.5/100的位置的距离）
        rho = self.getlocalDensity(dis)
        delta = self.computDelta(rho, dis)
        centers = self.identifyCenters(rho, delta)
        cores = []
        for i in range(self.size):
            if centers[i] != 0:
                cores.append(i)
        result = self.assignDataPoint(dis, rho, centers)
        fmi = lambda x, y: fowlkes_mallows_score(x, y)
        nmi = lambda x, y: normalized_mutual_info_score(x, y)
        ari = lambda x, y: adjusted_rand_score(x, y)
        print("%.3f,%.3f,%.3f" % (nmi(self.label, result), ari(self.label, result), fmi(self.label, result)))



    def Get_distance(self, points):
        dis = np.zeros((self.size, self.size))
        dist = []
        for i in range(self.size):
            for j in range(i + 1, self.size):
                dd = np.linalg.norm(points[i, :] - points[j, :])
                dis[i, j] = dd
                dis[j, i] = dd
                dist.append(dd)
        return dis, dist  ##ll是距离列表,dist是距离矩阵

    # 4 compute rho density
    def getlocalDensity(self, dist):
        rho = np.zeros(self.size)
        for i in range(self.size - 1):
            acc = 1
            acc_ = 1
            for j in range(self.size):
                yx = np.exp(-np.power(dist[i,j] / self.dc , 2))
                acc_ *= (1 - 1 / (1 + np.exp(-yx)))
                acc = acc * 1 / (1 + np.exp(-yx))
            rho[i] = acc / (acc_ + acc)
        print(rho)
        return rho

        # 5 compute Delta distance

    def computDelta(self, rho, dist):
        delta = np.ones((self.size, 1)) * self.MAX
        maxDensity = np.max(rho)  # 找到最大密度的点
        for i in range(self.size):
            if rho[i] < maxDensity:
                for j in range(self.size):
                    if rho[j] > rho[i] and dist[i][j] < delta[i]:
                        delta[i] = dist[i][j]
            else:
                delta[i] = 0.0
                for j in range(self.size):
                    if dist[i][j] > delta[i]:
                        delta[i] = dist[i][j]
        return delta

    # 6 identify cluster centers
    def identifyCenters(self, rho, delta):
        self.pf.printRhoDelta(rho,delta)
        # self.fo.writeData(centers, self.fileurl + 'centers.csv')
        return centers

        # 7 assign the remaining points to the corresponding cluster center

    # 7 assign the remaining points to the corresponding cluster center
    def assignDataPoint(self, dist, rho, result):
        for i in range(self.size):
            dist[i][i] = self.MAX

        for i in range(self.size):
            if result[i] == 0:
                result[i] = self.nearestNeighbor(i, dist, rho, result)
            else:
                continue
        return result

    # 8 Get the nearest neighbor with higher rho density for each point
    def nearestNeighbor(self, index, dist, rho, result):
        dd = self.MAX
        neighbor = -1
        for i in range(self.size):
            if dist[index, i] < dd and rho[index] < rho[i]:
                dd = dist[index, i]
                neighbor = i
        self.neigh[index] = neighbor
        if result[neighbor] == 0:  # 若没找到，则递归寻找
            result[neighbor] = self.nearestNeighbor(neighbor, dist, rho, result)
        return result[neighbor]


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    iodpc = IO_DPC()
    iodpc.runAlgorithm()
