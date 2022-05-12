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

class BPEC:
    def __init__(self):
        self.K = 75
        self.alpha0 = 1 / self.K
        self.alpha = 1
        self.beta = 2
        self.delta = 0.2
        self.eps = 10**(-3)
        self.q = 0.9
        self.fouci = FileOperatoruci()
        self.pf = PrintFigures()
        self.fo = FileOperator()
        # self.data,self.label = self.fouci.readWine("../dataset/wine.data")
        # self.data,self.label = self.fouci.readIris("../dataset/iris.data")
        self.data, self.label = self.fo.readFourclass("../dataset/fourclass.csv")#pair=2,K=75,q=0.9,delta = 0.2
        # self.data = self.fo.readButterfly("../dataset/butterfly.csv")
        min_max_scaler = preprocessing.MinMaxScaler()
        self.data = min_max_scaler.fit_transform(self.data)
        self.size,self.dim = self.data.shape
        self.c = 0
        self.pair_num = 2

    def Run(self):
        dis = self.calculate_Distance()
        KNN = self.calculate_Nearest_Neighbors(dis)
        gamma = self.calculate_Gamma(KNN, dis)
        belief = self.calculate_Belief(KNN, gamma, dis)
        delta_distance = self.calculate_Delta_distance(belief, dis)
        self.pf.printBPEC(belief, delta_distance)
        centers = self.identifyCenters(belief, delta_distance)
        print(centers)
        centers_point = self.data[centers]
        focal = self.make_focal_set("simple")#初始化焦点集，只有0和1和全集
        f = len(focal)
        M_all = self.focal_iterations(focal,centers_point)


        #添加新的集合后再计算
        similarity = self.calculate_similarity(M_all,focal,f)
        pairs = self.calculate_pairs(similarity)
        focal = self.make_focal_set(type = "pairs",pairs = pairs)
        M_all = self.focal_iterations(focal, centers_point)
        pro = self.calculate_pro(M_all, focal)
        result = self.get_result(pro)
        print(result)
        fmi = lambda x, y: fowlkes_mallows_score(x, y)
        nmi = lambda x, y: normalized_mutual_info_score(x, y)
        ari = lambda x, y: adjusted_rand_score(x, y)
        print("%.3f,%.3f,%.3f" % (
            nmi(self.label, result), ari(self.label, result), fmi(self.label, result)))



    #求分位数
    def quantile_exc(self, data, n):
        dic = {}
        a = 1
        data = list(data)
        for i in data:
            dic[a] = i
            a = a + 1
        value = ((a - 1) * n)
        return dic[math.ceil(value)]

    # 计算距离矩阵
    def calculate_Distance(self):
        dis = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(i + 1, self.size):
                dd = np.linalg.norm(self.data[i] - self.data[j])
                dis[i, j] = dd
                dis[j, i] = dd
        return dis

    def calculate_Nearest_Neighbors(self, dis):
        dis_sort = np.argsort(dis, axis=1)
        KNN = dis_sort.tolist()
        for i in range(self.size):
            KNN[i].remove(i)
        KNN = np.array(KNN)
        KNN = KNN[:, :self.K]
        return KNN

    # 计算每个点的gamma值
    def calculate_Gamma(self, KNN, dis):
        gamma = np.zeros(self.size)
        for i in range(self.size):
            # gamma[i] = 1 / np.percentile(dis[i,KNN[i]],self.q*100)
            gamma[i] = 1 / self.quantile_exc(dis[i, KNN[i]], self.q)
        return gamma

    # 计算每个点成为聚类中心的置信度
    def calculate_Belief(self, KNN, gamma, dis):
        belief = np.zeros(self.size)
        for i in range(self.size):
            belief_i = 1
            for j in KNN[i]:
                belief_i = belief_i * (1 - self.alpha0 * np.exp(-(np.power(gamma[j], 2)) * (np.power(dis[i, j], 2))))
            belief[i] = 1 - belief_i
        return belief

    # 计算delta距离
    def calculate_Delta_distance(self, belief, dis):
        delta_distance = np.zeros(self.size)
        max_belief = np.max(belief)
        for i in range(self.size):
            if belief[i] != max_belief:
                min_dis = np.inf
                for j in range(self.size):
                    if belief[j] > belief[i] and dis[i, j] < min_dis:
                        min_dis = dis[i, j]
                delta_distance[i] = min_dis
        delta_distance[np.where(belief == max_belief)[0]] = np.max(delta_distance)
        return delta_distance

    # 识别聚类中心
    def identifyCenters(self, belief, delta_distance):
        #iris
        thRho = 0.4
        thoDelta = 0.45
        #wine
        # thRho = 0.4
        # thoDelta = 0.8
        #fourclass
        thoDelta = 0.2
        thRho = 0.5
        centers = []
        cNum = 1
        for i in range(self.size):
            if belief[i] > thRho and delta_distance[i] > thoDelta:  # 只有大于thDel和thRho的点才能看成是中心
                centers.append(i)
                cNum += 1
        self.c = cNum - 1
        return centers

    #构造焦点集合,每一行为一个焦点集，里面包含所属的聚类中心，例如第3行里有第一个聚类和第二个聚类。
    def make_focal_set(self,type,Omega=True,pairs=None):
        if type == 'full':#也就是全集，有2^c个子集
            length = 2 ** self.c
            focal = np.zeros((length,self.c))
            for i in range(length):
                ten_to_two = bin(i)
                length = len(ten_to_two)
                for j in range(length-1,1,-1):
                    if ten_to_two[j] == '1':
                        focal[i,length - j - 1] = 1
        else:
            focal = np.concatenate((np.zeros((1,self.c)),np.eye(self.c)),axis=0)#把空集和独立子集拼接起来
            if type == 'pairs':#
                if pairs == None:#由于没指定，所以两两类都放入基础集
                    for i in range(self.c-1):
                        for j in range(i+1,self.c):
                            f = np.zeros((1,self.c))
                            f[0,i] = 1
                            f[0,j] = 1
                            focal = np.concatenate((focal,f),axis=0)
                else:#将指定的类对放入基础集里
                    length = len(pairs)
                    for i in range(length):
                        f = np.zeros((1,self.c))
                        f[0,pairs[i]] = 1
                        focal = np.concatenate((focal,f),axis=0)
            # if Omega and not((type == 'pairs') and (self.c==2)) and not((type=='simple') and(self.c==1)):
            #     focal = np.concatenate((focal,np.ones((1,self.c))),axis=0)#添加一个全集,这是可以选择的,fourclass里没有全集
        return focal

    #更新隶属度矩阵，D是size * （T-1）
    def calculate_M(self,f,centers_many,card,D):
        M = np.zeros((self.size,f-1))#Mij,也就是i与第j个焦点集的隶属度
        M0 = np.zeros((self.size,1))#属于空集的隶属度
        for i in range(self.size):
            sum_i_l = 0
            pk = 0
            for l in range(f-1):
                #若点i就是第j个焦点集的中心，则他与第j个焦点集的隶属度为1
                if all(self.data[i] == centers_many[l]):
                    pk = l+1
            if pk == 0:
                for l in range(1,f):
                    sum_i_l += np.power(card[l], -self.alpha / (self.beta-1)) * np.power(D[i,l-1], -2 / (self.beta-1))
                for j in range(f-1):
                    M[i,j] = np.power(card[j+1], -self.alpha / (self.beta-1)) * np.power(D[i,j], -2 / (self.beta-1)) / (sum_i_l + np.power(self.delta, -1/(self.beta-1)))
            else:
                M[i,pk-1] = 1
            M0[i] = 1 - np.sum(M[i])
        M_all = np.concatenate((M0,M),axis=1)
        return M0,M,M_all

    #计算损失函数值,f为焦点集的个数，card是每个焦点集的基数，D是每个点和非空焦点集中心的距离，size * （T-1）
    def calculate_JBPEC(self,f,card,D,M0,M):
        JBPEC = 0
        for i in range(self.size):
            for j in range(f - 1):
                JBPEC += np.power(card[j + 1], self.alpha) * np.power(M[i, j], self.beta) * np.power(D[i, j], 2)
            JBPEC += self.delta * np.power(M0[i, 0], self.beta)
        return JBPEC

    #计算每个点和集合重心之间的距离，其中f是除了空集之外的焦点集的个数(T-1)，focal为焦点集合(T*c)，S是权重距离矩阵(c*dim*dim)
    #center_many为焦点集合的中心，(T-1)*dim*dim,card是每个焦点集包含的聚类中心个数，T*1
    def calculate_D(self,f,focal,S,centers_many,card):
        D = np.zeros((self.size,f-1))
        for j in range(f-1):
            for i in range(self.size):
                sum_ikj = np.zeros((self.dim,1))
                for k in range(self.c):
                    if focal[j+1][k] == 1:#非空集
                        sum_ikj += np.dot(S[k], (self.data[i] - centers_many[j]).reshape(self.dim,1))
                D[i,j] = np.dot((self.data[i] - centers_many[j]),sum_ikj) / card[j+1]
        return D

    #更新S矩阵,M所有点对非空集合的隶属度，为size*(T-1),centers_many为非空焦点集的中心,为(T-1)*dim,card是每个焦点集的包含聚类中心数

    def calculate_S(self,M,centers_many,f,card,focal):
        tank = np.zeros((self.c,self.dim,self.dim))
        S = np.zeros((self.c,self.dim,self.dim))
        for k in range(self.c):
            temp = np.zeros((self.dim,self.dim))
            for i in range(self.size):
                for j in range(f-1):
                    if focal[j+1,k] == 1:
                        temp += np.power(card[j+1], self.alpha-1) * np.power(M[i,j],self.beta) * np.dot((self.data[i] - centers_many[j]).reshape(self.dim,1),(self.data[i] - centers_many[j]).reshape(1,self.dim))
            tank[k,:] = temp.copy()
        for i in range(self.c):
            S[i,:] = (np.power(np.linalg.det(tank[i]), 1 / self.dim) * np.linalg.inv(tank[i])).copy()
        return S


    def focal_iterations(self,focal,centers_point):
        f = len(focal)  # 焦点集的长度
        card = np.sum(focal, axis=1)#每个焦点集里有多少个聚类
        centers_many = np.zeros((f-1,self.dim))#每个子集里的聚类中心重心,由于空集不需要重心，所以只有f-1个，但是焦点集有f个
        iter = 0
        S = np.zeros((self.c,self.dim,self.dim))
        #初始化距离权重矩阵
        for i in range(self.c):
            for j in range(self.dim):
                S[i,j,j] = 1

        # 初始化每个子集里的聚类中心
        for i in range(1,f):
            focal_i = focal[i]
            sum_i = np.zeros(self.dim)
            num = 0
            for j in range(self.c):
                if focal_i[j] == 1:
                    sum_i += centers_point[j]
                    num += 1
            centers_many[i-1,:] = sum_i / num

        # D = self.calculate_D(f,focal,S,centers_many,card)

        #计算每个点到集合的隶属度
        # M0,M,M_all = self.calculate_M(f,centers_many,card,D)

        #开始迭代处理
        # S = self.calculate_S(M,centers_many,f,card,focal)
        M_all = np.zeros((self.size,f))
        while True:
            iter = iter + 1
            # 计算到中心的欧式距离
            M_last = M_all.copy()
            D = self.calculate_D(f, focal, S, centers_many, card)
            M0, M, M_all = self.calculate_M(f, centers_many, card, D)
            # 计算目标函数的损失值

            JBPEC = self.calculate_JBPEC(f, card, D, M0, M)
            dis = 0
            for i in range(f):
                dis += np.linalg.norm(M_all[i] - M_last[i])
            if dis < self.eps:
                break
            S = self.calculate_S(M, centers_many, f, card, focal)
        return M_all

    #计算两两聚类之间的相似度
    def calculate_similarity(self,M_all,focal,f):
        similarity = np.zeros((self.c,self.c))
        pl = np.zeros((self.size,self.c))
        for i in range(self.size):
            for j in range(self.c):
                for k in range(f):
                    if focal[k,j] == 1:
                        pl[i,j] += M_all[i,k]
        for j in range(self.c):
            for l in range(self.c):
                for i in range(self.size):
                    similarity[j,l] += pl[i,j] * pl[i,l]
        return similarity

    #计算要新增的焦点集合(只测试过近邻为1的)
    def calculate_pairs(self,similarity):
        arg_sort_s = np.argsort(similarity)[::-1].tolist()
        for i in range(self.c):
            arg_sort_s[i].remove(i)
        arg_sort_s = np.array(arg_sort_s)
        sort_index = arg_sort_s[:,:self.pair_num]
        pairs = []
        for k in range(self.pair_num):
            for i in range(self.c):
                for j in range(i,self.c):
                    if j in sort_index[i] and i in sort_index[j] and [i,j] not in pairs and [j,i] not in pairs:
                        pairs.append([i,j])
        return pairs

    def calculate_pro(self,M_all,focal):
        pro = np.zeros((self.size,self.c))
        f = len(focal)
        pl = np.zeros((self.size,self.c))
        for i in range(self.size):
            for j in range(self.c):
                for k in range(f):
                    if focal[k, j] == 1:
                        pl[i, j] += M_all[i, k]
        for i in range(self.size):
            for k in range(self.c):
                pro[i,k] = pl[i,k] / (np.sum(pl[i]))
        return pro

    def get_result(self,pro):
        result = np.zeros(self.size)
        for i in range(self.size):
            result[i] = np.argmax(pro[i])
        return result



if __name__ == "__main__":
    bpec = BPEC()
    bpec.Run()