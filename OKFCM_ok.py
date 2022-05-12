import math
import numpy as np
import random
import time
import copy
import sys
from tools.FileOperatoruci import FileOperatoruci
from tools.FileOperator import FileOperator
from tools.PrintFigures import PrintFigures
from sklearn.metrics import fowlkes_mallows_score, normalized_mutual_info_score, adjusted_rand_score
sys.path.append("../..")

class OKFCM:
    def __init__(self):
        self.ns = 15
        self.m = 2
        self.c = 3
        self.eps = 0.00000001
        self.Max = 10000.0
        self.sigma = 150
        self.fouci = FileOperatoruci()
        self.fo = FileOperator()
        self.pf = PrintFigures()
        self.data , self.label = self.fouci.readIris("../dataset/iris.data")
        self.size , self.dim = self.data.shape

    def Run(self):
        remind_data = self.data.copy()
        sample_data,remind_data = self.load_sample_data(remind_data)
        W = self.Initial_W()
        first_U,first_V = self.calculate_WFCM(sample_data,W,[])
        print("-----------------------")
        last_data_length = len(sample_data)
        sum_V,sum_W = self.iter_all(last_data_length,remind_data,first_U,first_V,W)
        final_U,final_V = self.calculate_WFCM(sum_V,sum_W,[])#得到最终的全局中心
        final_U = self.extension(self.data,final_V)
        result = np.argmax(final_U,axis=1)
        fmi = lambda x, y: fowlkes_mallows_score(x, y)
        nmi = lambda x, y: normalized_mutual_info_score(x, y)
        ari = lambda x, y: adjusted_rand_score(x, y)
        print("%.3f,%.3f,%.3f" % (nmi(self.label, result), ari(self.label, result), fmi(self.label, result)))
        return

    #迭代结束
    def end_conditon(self,U, U_old):
        """
        结束条件。当U矩阵随着连续迭代停止变化时，触发结束
        """
        for i in range(0, len(U)):
            dummy = 0.0
            for j in range(0, len(U[0])):
                dummy += abs(U[i][j] - U_old[i][j]) ** 2
            if dummy > self.eps:
                return False
        return True

    def distance(self, point, center):
        """
        该函数计算2点之间的RBF核距离
        """
        if len(point) != len(center):
            return -1
        dummy = 0.0
        for i in range(0, len(point)):
            dummy += abs(point[i] - center[i]) ** 2
        dummy = math.exp((-dummy) / (2 * self.sigma * self.sigma))
        return dummy

    #获取采样数据
    def load_sample_data(self,data):
        if not isinstance(data,list):
            data = data.tolist()
        sample_data = random.sample(data,self.ns)
        # sample_data = data[:self.ns]
        for i in sample_data:
            data.remove(i)
        print("本次抽取的数据的为:",sample_data)
        print("剩余：" + str(len(data)))
        print("抽样数据加载完毕")
        print("在150个数据中随机抽样：" + str(self.ns) + "个数据")
        return sample_data,data

    #初始化权重矩阵
    def Initial_W(self):
        W = []
        for i in range(self.ns):
            W.append(1)
        return W

    #初始化隶属度矩阵
    def Initial_U(self,data):
        U = []
        for i in range(len(data)):
            current = []
            rand_sum = 0.0
            for j in range(self.c):
                dummy = random.randint(1,int(self.Max))
                # dummy = (i * j ** 2) + 1
                current.append(dummy)
                rand_sum += dummy
            for j in range(self.c):
                current[j] = current[j] / rand_sum
            U.append(current)
        return U

    #初始化中心，是不需要上一轮中心的介入
    def Initial_V(self,U,data):
        length = len(data)
        center = []
        for j in range(self.c):
            current_center = []
            for i in range(self.dim):
                dummy_sum_num = 0.0
                dummy_sum_dum = 0.0
                for k in range(length):
                    dummy_sum_num += (U[k][j] ** self.m) * data[k][i]

                    dummy_sum_dum += (U[k][j] ** self.m)

                current_center.append(dummy_sum_num / dummy_sum_dum)
            center.append(current_center)
        return center

    #计算中心时是需要上一轮中心的介入
    def calculate_V(self,U,data,W,center):
        length = len(data)
        distance_matrix = []
        new_center = []
        for i in range(length):
            current = []
            for j in range(self.c):
                current.append(self.distance(data[i], center[j]))
            distance_matrix.append(current)
        for j in range(self.c):
            current_center = []
            for i in range(self.dim):
                dummy_sum_num = 0.0
                dummy_sum_dum = 0.0
                for k in range(length):
                    dummy_sum_num += (U[k][j] ** self.m) * data[k][i] * W[k] * distance_matrix[k][j]

                    dummy_sum_dum += (U[k][j] ** self.m) * W[k] * distance_matrix[k][j]

                current_center.append(dummy_sum_num / dummy_sum_dum)
            new_center.append(current_center)
        return new_center

    def calculate_U(self, data, center, U):
        length = len(data)
        distance_matrix = []
        for i in range(length):
            current = []
            for j in range(self.c):
                current.append(self.distance(data[i], center[j]))
            distance_matrix.append(current)
        for j in range(self.c):
            for i in range(length):
                flag = 0
                dummy = 0.0
                for k in range(self.c):
                    if (1 - distance_matrix[i][k]) == 0:  # 如果是中心，则把隶属度设为1，其余为0
                        U[i] = [0 for i in range(self.c)]
                        U[i][k] = 1
                        flag = 1
                        break
                    dummy += (1 - distance_matrix[i][k]) ** (-1 / (self.m - 1))
                if flag == 0:
                    U[i][j] = ((1 - distance_matrix[i][j]) ** (-1 / (self.m - 1))) / dummy
        return U

    def calculate_WFCM(self,data,W,V):
        U = self.Initial_U(data)
        if V == []:
            V = self.Initial_V(U,data)
            iter = 1
        else:
            iter = 0
        while True:
            iter += 1
            U_old = copy.deepcopy(U)
            U = self.calculate_U(data,V,U)
            V = self.calculate_V(U,data,W,V)
            if self.end_conditon(U,U_old):
                break
        print("本次迭代结束")
        return U,V

    def end_conditon(self,U, U_old):
        """
        结束条件。当U矩阵随着连续迭代停止变化时，触发结束
        """
        for i in range(0, len(U)):
            dummy = 0.0
            for j in range(0, len(U[0])):
                dummy += abs(U[i][j] - U_old[i][j]) ** 2
            if dummy > self.eps:
                return False
        return True
    def iter_all(self,last_data_length,remind_data,U,V,W):
        sum_V = copy.deepcopy(V)
        sum_W = []
        for i in range(self.c):#第一次得到的U
            tmp = 0
            for j in range(last_data_length):
                tmp += U[j][i]
            sum_W.append(tmp)
        for i in range(1,int(self.size / self.ns)):
            print("第" + str(i) + "次采样")
            sample_data,remind_data = self.load_sample_data(remind_data)
            U,V = self.calculate_WFCM(sample_data,W,V)
            sum_V += V
            length = len(sample_data)
            for i in range(self.c):
                tmp = 0
                for j in range(length):
                    tmp += U[j][i]
                sum_W.append(tmp)
        print("------------------")
        return sum_V,sum_W

    def extension(self,data,V):
        dis = []
        for i in range(self.size):
            current = []
            for j in range(self.c):
                current.append(self.distance(data[i],V[j]))
            dis.append(current)
        U = [[0] * self.c for i in range(self.size)]
        for j in range(self.c):
            for i in range(self.size):
                dummy = 0.0
                for k in range(self.c):
                    dummy += (1 - dis[i][k]) ** (-1 / (self.m - 1))
                U[i][j] = ((1 - dis[i][j]) ** (-1 / self.m - 1)) / dummy
        print("拓展到整个数据集上")
        print("硬聚类化 U")
        U = self.normalise_U(U)
        return U

    def normalise_U(self,U):
        """
        在聚类结束时使U模糊化。每个样本的隶属度最大的为1，其余为0，也就是变成硬分类
        """
        for i in range(0, len(U)):
            maximum = max(U[i])
            for j in range(0, len(U[0])):
                if U[i][j] != maximum:
                    U[i][j] = 0
                else:
                    U[i][j] = 1
        return U


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    okfcm = OKFCM()
    okfcm.Run()

