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
sys.setrecursionlimit(3000)
class FDPC:


    def __init__(self,K,T,sigma):
        self.fo = FileOperator()
        self.fouci = FileOperatoruci()
        self.pf = PrintFigures()
        self.data, self.label = self.fo.readDatawithLabel2("../../datasets/R15.txt")#15个
        # self.name = "R15"
        # self.data,self.label = self.fo.readDatawithLabel("../../datasets/data/Synthetic/diamond9.csv")#9个
        # self.name = "diamond9"
        ##self.data,self.label = self.fo.readDatawithLabel("../../datasets/data/Synthetic/twenty.arff")#
        # self.name = "twenty"
        # self.data, self.label = self.fo.readDatawithLabel2("../../datasets/flame.txt")#
        # self.name = "flame"
        # self.data, self.label = self.fo.readDatawithLabel("../../datasets/fourty.arff")#
        # self.name = "fourty"
        # self.data, self.label = self.fo.readDatawithLabel("../../datasets/data/Synthetic/elliptical_10_2.arff")#
        # self.name = "elliptical"
        # self.data, self.label = self.fo.readData3D("../../datasets/data/Synthetic/hepta.arff")#
        # self.name = "hepta"
        # self.data, self.label = self.fo.readDatawithLabel("../../datasets/data/Synthetic/sizes3.arff")#
        # self.name = "size3"

        # self.data, self.label = self.fo.readDatawithLabel("../../datasets/data/Synthetic/sizes4.arff")#
        # self.name = "size4"
        # self.data, self.label = self.fo.readDatawithLabel("../../datasets/data/Synthetic/sizes5.arff")#
        # self.name = "size5"

        # self.pf.printScatter(self.data)
        # self.pf.print_Dcores_order(self.data)
        # self.data,self.label = self.fouci.readIris("../../uci datasets/iris.data")#3个
        # self.data,self.label = self.fouci.readSeed("../../uci datasets/seeds_dataset.txt")#3个
        # self.data, self.label = self.fouci.readDivorce("../../uci datasets/divorce.csv")#2个
        # self.data,self.label = self.fouci.readLymphography("../../uci datasets/lymphography.data")#4个
        # self.data,self.label = self.fouci.readParkinsons("../../uci datasets/parkinsons.data")#2个
        # self.data, self.label = self.fouci.readSegmentation(
        #     "../../uci datasets/segmentation.data")  # 7个
        # self.data,self.label = self.fouci.readZoo("../../uci datasets/zoo.data")#7个
        # self.data , self.label = self.fouci.readWine("../../uci datasets/wine.data")#3个
        self.size,self.dim = self.data.shape
        self.neigh = np.zeros(self.size)
        self.w = 0.1
        self.T = T#(0.01,0.6,0.01)
        self.sigma = sigma#(0.1,20,0.1)
        self.K = K#(2,20,1)
        self.neigh = np.zeros(self.size)

    def runAlgorithm(self):
        # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        # points = min_max_scaler.fit_transform(points)
        fcl = []
        dis= self.Get_distance(self.data)
        Y = self.calculate_Y(dis)
        rho = self.getlocalDensity(Y,dis)
        delta = self.computDelta(rho, dis)
        centers = self.identifyCenters(rho, delta)
        cores = []
        for i in range(self.size):
            if centers[i] != 0:
                cores.append(i)
        # result = self.assignDataPoint(dis, rho, centers)
        # fmi = lambda x, y: fowlkes_mallows_score(x, y)
        # nmi = lambda x, y: normalized_mutual_info_score(x, y)
        # ari = lambda x, y: adjusted_rand_score(x, y)
        # print("%.3f,%.3f,%.3f" % (nmi(self.label, result), ari(self.label, result), fmi(self.label, result)))



    def Get_distance(self, points):
        dis = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(i + 1, self.size):
                dd = np.linalg.norm(points[i, :] - points[j, :])
                dis[i, j] = dd
                dis[j, i] = dd
        return dis  ##ll是距离列表,dist是距离矩阵

    def calculate_Y(self,dis):
        Y = np.zeros((self.size,self.size))
        for i in range(self.size-1):
            for j in range(i,self.size):
                Y[i,j] = np.exp(-np.power(dis[i,j],2) / (2 * np.power(self.sigma,2)))
                Y[j,i] = np.exp(-np.power(dis[i,j],2) / (2 * np.power(self.sigma,2)))
        return Y


    # 4 compute rho density
    def getlocalDensity(self, Y,dis):
        rho = np.zeros(self.size)
        dis_index = np.argsort(dis,axis=1).tolist()
        for i in range(self.size):
            dis_index[i].remove(i)
        dis_index = np.array(dis_index)
        for i in range(self.size):
            F_Y = 1 - np.exp(-Y[i,dis_index[i,:self.K]])
            cumpro_i = np.cumprod(F_Y)[self.K-1]
            sum_i = np.sum(F_Y)
            rho[i] = (sum_i - (2 - self.w) * cumpro_i) / (1 - (1 - self.w) * cumpro_i)
        return rho

        # 5 compute Delta distance

    def computDelta(self, rho, dist):
        delta = np.ones((self.size, 1)) * np.inf
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
        max_rho = np.max(rho)
        min_rho = np.min(rho)
        max_delta = np.max(delta)
        min_delta = np.min(delta)
        rho_delta = np.zeros(self.size)
        for i in range(self.size):
            rho[i] = (rho[i] - min_rho) / (max_rho - min_rho)
            delta[i] = (delta[i] - min_delta) / (max_delta - min_delta)
            rho_delta[i] = rho[i] * delta[i]
        centers = np.zeros(self.size, dtype=np.int)
        cNum = 1
        rho_delta_sort = np.sort(rho_delta)[::-1]
        rho_delta_index = np.argsort(rho_delta)[::-1]
        for i in range(self.size-1):
            if rho_delta_sort[i] - rho_delta_sort[i+1] > self.T:
                centers[rho_delta_index[i]] = cNum
                cNum += 1
        print("Number of cluster centers: ", cNum - 1)
        # self.fo.writeData(centers, self.fileurl + 'centers.csv')
        return centers

        # 7 assign the remaining points to the corresponding cluster center

    # 7 assign the remaining points to the corresponding cluster center
    def assignDataPoint(self, dist, rho, result):
        for i in range(self.size):
            dist[i][i] = np.inf

        for i in range(self.size):
            if result[i] == 0:
                result[i] = self.nearestNeighbor(i, dist, rho, result)
            else:
                continue
        return result

    # 8 Get the nearest neighbor with higher rho density for each point
    def nearestNeighbor(self, index, dist, rho, result):
        dd = np.inf
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
    for k in range(2,20):
        for T in np.arange(0.1,0.6,0.05):
            for sigma in np.arange(1,20,1):
                print(k,T,sigma)
                fdpc = FDPC(k,T,sigma)
                fdpc.runAlgorithm()
