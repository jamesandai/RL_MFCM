'''
Suporting printer services
Created on 2017-9-27
@author: Jianguo Chen
'''
import os

import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.patches  import Circle
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D

class PrintFigures:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    #1 绘制点的曲线图
    def printPolt2(self,ylist):
        plt.figure()

        for i in range(len(ylist)):
            plt.plot(i, ylist[i], marker = '.')
        plt.xlabel('x'), plt.ylabel('y')
        plt.show()
    
    #2 绘制曲线图
    def printPolt3(self,ylist):
        plt.figure()
        plt.plot(ylist)
        plt.xlabel('x'), plt.ylabel('y')
        plt.show()
    
    #3 绘制散点图，无色
    def printScatter(self,points):
        plt.figure()
        for i in range(len(points)):
            plt.plot(points[i][0], points[i][1],color='#0049A4', marker = '.')
            plt.text(points[i][0], points[i][1], s=str(i+1))
        plt.xlabel('x'), plt.ylabel('y')
        plt.title("无颜色的散点图", y=-0.2)
        y_locator = MultipleLocator(5)
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_locator)
        plt.show()
 
    #4 绘制散点图，根据类别填色
    def printScatter_Color_order(self,points,label):
        colors = self.getRandomColor()          
        fig = plt.figure()
        ax = fig.add_subplot(111) #1*1网格里第一个子图
        for i in range(len(points)):
            index = int(label[i])
            plt.plot(points[i][0], points[i][1], color = colors[index], marker = '.')
            plt.text(points[i][0],points[i][1],s=str(i))
        # xmin, xmax = plt.xlim()   # 返回x上的最小值和最大值
        # ymin, ymax = plt.ylim()   # 返回y上的最小值和最大值
        # plt.xlim(xmin=int(xmin* 1.0), xmax=int(xmax *1.1))  #set the axis range
        # plt.ylim(ymin = int(ymin * 1.0), ymax=int(ymax * 1.1))
        #
        # xmajorLocator   = MultipleLocator(4) #x主刻度标签设置为4的倍数
        # ax.xaxis.set_major_locator(xmajorLocator)  #自动线性调整
        plt.xticks(fontsize = 17) 
        plt.yticks(fontsize = 17)
        plt.title("有颜色的散点图",y=-0.2)
        plt.show()



    # def printScatter_Color(self, points, label,flag):
    def printScatter_Color(self, points, label,k,name):
        colors = self.getRandomColor()
        fig = plt.figure()
        if points.shape[1] == 2:
            for i in range(len(points)):
                index = int(label[i])
                # if index == 0:
                #     plt.scatter(points[i][0], points[i][1], color=colors[index], edgecolors='b', marker='*')
                # else:
                plt.plot(points[i][0], points[i][1], color=colors[index], marker='.')
        else:
            ax = Axes3D(fig)
            for i in range(len(points)):
                index = int(label[i])
                if index == 0:
                    ax.scatter(points[i][0], points[i][1],points[i][2], color=colors[index], edgecolors='b', marker='*')
                else:
                    ax.scatter(points[i][0], points[i][1],points[i][2], color=colors[index], marker='.')
        plt.title("有颜色的散点图", y=-0.2)
        if k != 0:
            if not os.path.exists('../图片/'+name+'/result'):
                os.makedirs('../图片/'+name+'/result')
            plt.savefig(os.path.join('../图片/'+name+'/result', str(k)))
        plt.close('all')
        # plt.show()
    def printScatter_Color_out(self, points, label,outliers):
        colors = self.getRandomColor()
        fig = plt.figure()
        ax = fig.add_subplot(111)  # 1*1网格里第一个子图
        for i in range(len(points)):
            if i in outliers:
                continue
            index = int(label[i])
            plt.plot(points[i][0], points[i][1], color=colors[index], marker='.')
        plt.show()
    #5 绘制散点图，有颜色和类别的标记,实例图在左下角
    def printScatter_Color_Marker(self,points,label):
        colors = self.getRandomColor()  
        markers = self.getRandomMarker() 
        cNum = int(np.max(label))  #Number of clusters
        print(cNum)
        for j in range(cNum):
            idx = np.where(label==j)  
            plt.scatter(points[idx,0], points[idx,1], color =  colors[j%len(colors)], label=('C'+str(j)), marker = markers[j%len(markers)], s = 20)  
        
        plt.xticks(fontsize = 13)
        plt.yticks(fontsize = 13)
        plt.legend(loc = 'lower left')
        plt.title("有颜色有实例的散点图", y=-0.2)
        plt.show()    
      
    #6 用圆圈绘制散点图 (coloring and marking with label)
    #  为每个点设置一个具有指定半径的圆
    #  输入：points，label，rs：每个数据点需要的圆的半径
    def printScatterCircle(self,points,label,rs):
        colors = self.getRandomColor()  
        markers = self.getRandomMarker() 
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cNum = np.max(label)
        for j in range(cNum+1):
            print("j=",j)
            print("range=",range(cNum))
            idx = np.where(label==j)
            print("idx:",idx)   
            plt.scatter(points[idx,0], points[idx,1], color =  colors[j%len(colors)], label=('C'+str(j)), marker = markers[j%len(markers)], s = 30)  
        #plt.xlabel('x'), plt.ylabel('y')
        for i in range(len(points)): 
            print("rs","i:",rs[i])
            cir1 = Circle(xy = (points[i,0], points[i,1]), radius=rs[i], alpha=0.03)
            ax.add_patch(cir1)
            ax.plot(points[i,0], points[i,1], 'w')
        plt.legend(loc = 'best')
        plt.title("有颜色有实例的散点图", y=-0.2)
        plt.show()     
        
           
    #7 绘制带有图例的曲线图，实例图在左上方
    def printPoltLenged(self,points,label):
        colors = self.getRandomColor()  
        markers = self.getRandomMarker() 
        plt.figure()
        cNum = np.max(label) 
        for j in range(cNum+1):
            idx = np.where(label==j)  
            plt.scatter(points[idx,0], points[idx,1], color =  colors[j%len(colors)], label=('C'+str(j)), marker = markers[j%len(markers)], s = 30)  
        plt.legend(loc = 'upper left')
        plt.xticks(fontsize = 17)
        plt.yticks(fontsize = 17)
        plt.title("有颜色有实例的曲线图", y=-0.2)
        plt.show()
      
        
    #8 生成聚类决策图
    # X-axis: rho, Y-axis: delta
    def printRhoDelta(self,belief,delta):
        plt.plot(belief, delta, '.', MarkerSize=15)
        plt.xticks(fontsize = 17)
        plt.yticks(fontsize = 17)
        plt.xlabel('x', fontsize=17)
        plt.ylabel('y', fontsize=17)
        # plt.xlim(0.1, 0.9)
        # plt.ylim(0.1, 0.6)
        plt.xlabel('density'), plt.ylabel('delta')
        plt.show()

    def printBPEC(self,belief,delta):
        #plt.plot(delta, belief, '.', MarkerSize=15)
        length = len(belief)
        for i in range(length):
            plt.plot(delta[i], belief[i], color='b', marker='.')
            plt.text(delta[i],belief[i],str(i))
        plt.xticks(fontsize = 17)
        plt.yticks(fontsize = 17)
        plt.xlabel('x', fontsize=17)
        plt.ylabel('y', fontsize=17)
        # plt.xlim(0.1, 0.8)
        # plt.ylim(0.1, 0.6)
        plt.xlabel('delta'), plt.ylabel('belief')
        plt.show()

    def printRhoDelta2(self, rho, delta,throld,size):
        plt.figure()
        plt.plot(rho,delta)
        plt.plot([0, size], [throld, throld], color='r')
        plt.show()

    #RDMN三次最小生成树的图
    def print_three_rounds(self,points,neighbor):
        plt.figure()
        neighbor_length = len(neighbor)
        for i in range(neighbor_length):
            a = neighbor[i][0]
            b = neighbor[i][1]
            plt.plot([points[a][0],points[b][0]],[points[a][1],points[b][1]],color='r')
        for i in range(len(points)):
            plt.plot(points[i][0], points[i][1], color='#0049A4', marker='o')
        plt.xlabel('x'), plt.ylabel('y')
        plt.title("领域图", y=-0.2)
        plt.show()

    def print_SubCluster_MST(self,points,neighbor):
        plt.figure()
        neighbor_length = len(neighbor)
        for i in range(neighbor_length):
            a = neighbor[i][0]
            b = neighbor[i][1]
            plt.plot([points[a][0],points[b][0]],[points[a][1],points[b][1]],color='r')
        for i in range(len(points)):
            plt.plot(points[i][0], points[i][1], color='#0049A4', marker='o')
        plt.xlabel('x'), plt.ylabel('y')
        plt.title("最小生成树图", y=-0.2)
        plt.show()

    def print_CFSFDP_line(self,points,label,neigh):
        plt.figure()
        length = len(points)
        colors = self.getRandomColor()
        for i in range(length):
            print(i,neigh[i])
            plt.plot([points[i][0],points[neigh[i]][0]],[points[i][1],points[neigh[i]][1]],color='r')
        for i in range(len(points)):
            plt.plot(points[i][0], points[i][1], color=colors[label[i]], marker='o')
    def print_MST_distance(self,points,neighbor):
        plt.figure()
        neighbor_length = len(neighbor)
        for i in range(neighbor_length):
            a = neighbor[i][0]
            b = neighbor[i][1]
            plt.plot([points[a][0],points[b][0]],[points[a][1],points[b][1]],color='r')
            # s = ''+str(a)+'+'+str(b) +"+"+ str(round(neighbor[i][2],1))
            s =str(round(neighbor[i][2], 2))
            plt.text((points[b][0] + points[a][0])/2,(points[a][1]+points[b][1])/2,s)
        for i in range(len(points)):
            plt.plot(points[i][0], points[i][1], color='#0049A4', marker='o')
        plt.xlabel('x'), plt.ylabel('y')
        plt.title("带距离的最小生成树图", y=-0.2)
        plt.show()
    # def print_Dcores_order(self,points,flag):
    def print_Dcores_order(self, points):
        fig = plt.figure()
        ax = fig.add_subplot(111)  # 1*1网格里第一个子图
        for i in range(len(points)):
            # if i not in Dcores:
            #     plt.plot(points[i][0], points[i][1], color='b', marker='.')
            plt.plot(points[i][0], points[i][1], color='b', marker='.')
            plt.text(points[i][0],points[i][1],str(i))
        xmin, xmax = plt.xlim()  # 返回x上的最小值和最大值
        ymin, ymax = plt.ylim()  # 返回y上的最小值和最大值
        # plt.xlim(xmin=int(xmin * 1.0), xmax=int(xmax * 1.1))  # set the axis range
        # plt.ylim(ymin=int(ymin * 1.0), ymax=int(ymax * 1.1))
        # xmajorLocator = MultipleLocator(4)  # x主刻度标签设置为4的倍数
        # ax.xaxis.set_major_locator(xmajorLocator)  # 自动线性调整
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.title("密度中心图", y=-0.2)
        # fig.savefig(flag + '2fig.png')
        plt.show()

    def print_Dcores_distance(self,points,cores,GD):
        plt.figure()
        length = len(cores)
        for i in range(length):
            arg_sort = np.argsort(GD[i,:])
            for j in arg_sort[:5]:
                if i == j:
                    continue
                else:
                    plt.plot([points[cores[i]][0],points[cores[j]][0]],[points[cores[i]][1],points[cores[j]][1]],color='r')
            # s = ''+str(a)+'+'+str(b) +"+"+ str(round(neighbor[i][2],1))
                s =str(round(GD[i][j], 2))
                plt.text((points[cores[i]][0] + points[cores[j]][0])/2,(points[cores[i]][1]+points[cores[j]][1])/2,s)
        for i in range(length):
            plt.plot(points[cores[i]][0], points[cores[i]][1], color='#0049A4', marker='o')
            plt.text(points[cores[i]][0]-0.1, points[cores[i]][1]-0.1,''.join(str(i+1)))
        plt.xlabel('x'), plt.ylabel('y')
        plt.title("带距离的最小生成树图", y=-0.2)
        plt.show()
    def print_Dcores_Den(self,points,Dcores,Den):
        fig = plt.figure()
        ax = fig.add_subplot(111)  # 1*1网格里第一个子图
        for i in range(len(points)):
            # if i not in Dcores:
            #     plt.plot(points[i][0], points[i][1], color='b', marker='.')
            if i in Dcores:
                plt.plot(points[i][0], points[i][1], color='R', marker='.')
                plt.text(points[i][0],points[i][1],''.join(str(round(Den[i],1))))
        plt.show()
    def print_Den(self,points,Den):
        fig = plt.figure()
        ax = fig.add_subplot(111)  # 1*1网格里第一个子图
        for i in range(len(points)):
            plt.plot(points[i][0], points[i][1], color='R', marker='.')
            plt.text(points[i][0],points[i][1],''.join(str(round(Den[i],0))))
        xmin, xmax = plt.xlim()  # 返回x上的最小值和最大值
        ymin, ymax = plt.ylim()  # 返回y上的最小值和最大值
        plt.xlim(xmin=int(xmin * 1.0), xmax=int(xmax * 1.1))  # set the axis range
        plt.ylim(ymin=int(ymin * 1.0), ymax=int(ymax * 1.1))

        xmajorLocator = MultipleLocator(4)  # x主刻度标签设置为4的倍数
        ax.xaxis.set_major_locator(xmajorLocator)  # 自动线性调整
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.title("密度中心图", y=-0.2)
        plt.show()
    # def print_Dcores_line(self,points,Dcores,Rep,flag,outliers):
    def print_Dcores_line(self, points, Dcores, Rep,outliers):
        fig = plt.figure()
        ax = fig.add_subplot(111)  # 1*1网格里第一个子图
        for i in range(len(points)):
            if i not in Dcores:
                if i in outliers:
                    plt.plot(points[i][0], points[i][1], color='y', marker='')
                plt.plot(points[i][0], points[i][1], color='b', marker='.')
            if i in Dcores:
                plt.scatter(points[i][0], points[i][1], color='w', edgecolors='b', marker='*',s=500)
        for i in range(len(points)):
            if i not in outliers:
                plt.plot([points[i][0],points[Rep[i]][0]],[points[i][1],points[Rep[i]][1]],color='r')
        plt.title("密度中心图", y=-0.2)
        # fig.savefig(flag + '3fig.png')
        plt.show()

    def print_Dcores_center(self,points, Dcores):
        fig = plt.figure()
        ax = fig.add_subplot(111)  # 1*1网格里第一个子图
        for i in range(len(points)):
            if i in Dcores:
                plt.plot(points[i][0], points[i][1], color='R', marker='*')
            else:
                plt.plot(points[i][0], points[i][1], color='y', marker='.')
        xmin, xmax = plt.xlim()  # 返回x上的最小值和最大值
        ymin, ymax = plt.ylim()  # 返回y上的最小值和最大值
        plt.xlim(xmin=int(xmin * 1.0), xmax=int(xmax * 1.1))  # set the axis range
        plt.ylim(ymin=int(ymin * 1.0), ymax=int(ymax * 1.1))

        xmajorLocator = MultipleLocator(4)  # x主刻度标签设置为4的倍数
        ax.xaxis.set_major_locator(xmajorLocator)  # 自动线性调整
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.title("密度中心图", y=-0.2)
        plt.show()

    def print_fcm_center(self,points, centers,k,name):
        fig = plt.figure()  # 1*1网格里第一个子图
        if points.shape[1] == 2:
            ax = fig.add_subplot(111)
            for i in range(len(points)):
                plt.plot(points[i][0], points[i][1], color='y', marker='.')
            for i in range(len(centers)):
                plt.plot(centers[i][0],centers[i][1],color='r',marker='*')
            xmin, xmax = plt.xlim()  # 返回x上的最小值和最大值
            ymin, ymax = plt.ylim()  # 返回y上的最小值和最大值
            plt.xlim(xmin=int(xmin * 1.0), xmax=int(xmax * 1.1))  # set the axis range
            plt.ylim(ymin=int(ymin * 1.0), ymax=int(ymax * 1.1))

            xmajorLocator = MultipleLocator(4)  # x主刻度标签设置为4的倍数
            ax.xaxis.set_major_locator(xmajorLocator)  # 自动线性调整
            plt.xticks(fontsize=17)
            plt.yticks(fontsize=17)
            plt.title("密度中心图", y=-0.2)
        else:
            ax = Axes3D(fig)
            for i in range(len(points)):
                ax.scatter(points[i][0],points[i][1],points[i][2],color='y', marker='.')
            for i in range(len(centers)):
                ax.scatter(centers[i][0], centers[i][1],centers[i][2], color='r', marker='*')
        if not os.path.exists('../图片/'+name+'/centers'):
            os.makedirs('../图片/'+name+'/centers')
        plt.savefig(os.path.join('../图片/'+name+'/centers',str(k)))
        plt.close('all')
        # plt.show()


    def print_Rep_line(self, points, Rep):
        fig = plt.figure()
        ax = fig.add_subplot(111)  # 1*1网格里第一个子图
        for i in range(len(points)):
            plt.scatter(points[i][0], points[i][1], color='w', edgecolors='b', marker='.')
        for i in range(len(points)):
            start = points[i]
            end = points[Rep[i]]
            # plt.plot([points[i][0],points[Rep[i]][0]],[points[i][1],points[Rep[i]][1]],color='r')
            plt.arrow(start[0], start[1], (end[0] - start[0])*0.9, (end[1] - start[1])*0.9,
                      length_includes_head=False, head_width=0.02, lw=0.1,
                      color='r')
        # fig.savefig(flag + '3fig.png')
        plt.show()
    def print_Dcores_line_step2(self, points, Dcores, Rep):
        fig = plt.figure()
        ax = fig.add_subplot(111)  # 1*1网格里第一个子图
        for i in range(len(points)):
            plt.plot([points[i][0],points[Rep[i]][0]],[points[i][1],points[Rep[i]][1]],color='black')
        k = 1
        for i in range(len(points)):
            if i not in Dcores:
                plt.plot(points[i][0], points[i][1], color='blue', marker='.')
            if i in Dcores:
                plt.plot(points[i][0], points[i][1],'r*')
                plt.text(points[i][0]-0.2,points[i][1]-0.1,''.join(str(k)))
                k += 1
        plt.show()
    def print_Dcores_NND(self, points, Dcores, NND):
        fig = plt.figure()
        ax = fig.add_subplot(111)  # 1*1网格里第一个子图
        for i in range(len(Dcores)):
            for j in NND[i]:
                plt.plot([points[Dcores[i]][0],points[j][0]],[points[Dcores[i]][1],points[j][1]],color='green')
        k = 1
        for i in range(len(points)):
            if i not in Dcores:
                plt.plot(points[i][0], points[i][1], color='blue', marker='.')
            if i in Dcores:
                plt.plot(points[i][0], points[i][1],'r*')
                plt.text(points[i][0]-0.2,points[i][1]-0.1,''.join(str(k)))
                k += 1
        plt.show()
    def print_HC_line_step4(self, points, Dcores, Rep,HC):
        fig = plt.figure()
        ax = fig.add_subplot(111)  # 1*1网格里第一个子图
        for i in range(len(points)):
            plt.plot([points[i][0],points[Rep[i]][0]],[points[i][1],points[Rep[i]][1]],color='black')
        for i in range(len(points)):
            if i not in Dcores:
                plt.plot(points[i][0], points[i][1], color='blue', marker='.')
            if i in Dcores:
                plt.plot(points[i][0], points[i][1],'r*')
        for i in range(len(HC)):
            if len(HC[i]) >= 2:
                Start = Dcores[HC[i][0]]
                for j in HC[i]:
                    if j == HC[i][0]:
                        continue
                    end = Dcores[j]
                    plt.plot([points[Start][0], points[end][0]], [points[Start][1], points[end][1]], color='limegreen')
        plt.show()
    def printScatter_Color_snndpc(self, points, label,center):
        colors = self.getRandomColor()
        fig = plt.figure()
        ax = fig.add_subplot(111)  # 1*1网格里第一个子图
        for i in range(len(points)):
            index = int(label[i])
            if i not in center:
                plt.plot(points[i][0], points[i][1], color=colors[index], marker='.')
            if i in center:
                # plt.plot(points[i][0], points[i][1],marker='*',MarkerSize=10)
                plt.scatter(points[i][0],points[i][1],color='w',edgecolors='b',marker='*')
        xmin, xmax = plt.xlim()  # 返回x上的最小值和最大值
        ymin, ymax = plt.ylim()  # 返回y上的最小值和最大值
        plt.xlim(xmin=int(xmin * 1.0), xmax=int(xmax * 1.1))  # set the axis range
        plt.ylim(ymin=int(ymin * 1.0), ymax=int(ymax * 1.1))

        xmajorLocator = MultipleLocator(4)  # x主刻度标签设置为4的倍数
        ax.xaxis.set_major_locator(xmajorLocator)  # 自动线性调整
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.title("有颜色的散点图", y=-0.2)
        plt.show()

    #看离群点
    def printScatter_Color_my(self, points,outliers):
        colors = self.getRandomColor()
        fig = plt.figure()
        ax = fig.add_subplot(111)  # 1*1网格里第一个子图
        for i in range(len(points)):
            if i not in outliers:
                plt.plot(points[i][0], points[i][1], color='#0049A4', marker='.')
            if i in outliers:
                plt.scatter(points[i][0],points[i][1],color='w',edgecolors='b',marker='*')
        xmin, xmax = plt.xlim()  # 返回x上的最小值和最大值
        ymin, ymax = plt.ylim()  # 返回y上的最小值和最大值
        plt.xlim(xmin=int(xmin * 1.0), xmax=int(xmax * 1.1))  # set the axis range
        plt.ylim(ymin=int(ymin * 1.0), ymax=int(ymax * 1.1))

        xmajorLocator = MultipleLocator(4)  # x主刻度标签设置为4的倍数
        ax.xaxis.set_major_locator(xmajorLocator)  # 自动线性调整
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.title("有颜色的散点图", y=-0.2)
        plt.show()

    def printScatter_Color_my2(self,points,Den):
        fig = plt.figure()
        ax = fig.add_subplot(111)  # 1*1网格里第一个子图
        for i in range(8000,len(points)):
            plt.plot(points[i][0], points[i][1], color='#0049A4', marker='.')
            plt.text(points[i][0], points[i][1], ''.join(str(round(Den[i],0))))
        xmin, xmax = plt.xlim()  # 返回x上的最小值和最大值
        ymin, ymax = plt.ylim()  # 返回y上的最小值和最大值
        plt.xlim(xmin=int(xmin * 1.0), xmax=int(xmax * 1.1))  # set the axis range
        plt.ylim(ymin=int(ymin * 1.0), ymax=int(ymax * 1.1))

        xmajorLocator = MultipleLocator(4)  # x主刻度标签设置为4的倍数
        ax.xaxis.set_major_locator(xmajorLocator)  # 自动线性调整
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.title("有颜色的散点图", y=-0.2)
        plt.show()
    def print_three_rounds_Den(self, points, neighbor,Den):
        plt.figure()
        neighbor_length = len(neighbor)
        for i in range(neighbor_length):
            a = neighbor[i][0]
            b = neighbor[i][1]
            plt.plot([points[a][0], points[b][0]], [points[a][1], points[b][1]], color='r')
        for i in range(len(points)):
            plt.plot(points[i][0], points[i][1], color='#0049A4', marker='o')
            s = '' + str(i) + '++' + str(round(Den[i],3))
            plt.text(points[i][0],points[i][1],s)
        plt.xlabel('x'), plt.ylabel('y')
        plt.title("带密度的邻域图", y=-0.2)
        plt.show()

    #输出每个集合里的密度分布
    def print_den_sort(self,cores,Den):
        plt.figure()
        length = len(Den)
        x = np.arange(length)
        for i in range(length):
            plt.plot(x[i],Den[i],'.',color='#0049A4')
            plt.text(x[i], Den[i], ''.join(str(cores[i])+"+"+str(round(Den[i],2))))
        plt.xlabel('x'),
        plt.ylabel('den')
        plt.title("密度图", y=-0.2)
        plt.show()


    #10 生成随机颜色
    def getRandomColor(self):
        R = list(range(256))  # np.arange(256)
        B = list(range(256))
        G = list(range(256))
        R = np.array(R) / 255.0
        G = np.array(G) / 255.0
        B = np.array(B) / 255.0
        # print(R)
        random.shuffle(R)
        random.shuffle(G)
        random.shuffle(B)
        colors = []
        for i in range(256):
            colors.append((R[i], G[i], B[i]))
        # colors = ['peru','deepskyblue','fuchsia','lightpink','lightslategrey','darkred','darkorange','blueviolet','darkslategrey','springgreen','darkcyan','indigo','crimson','mediumblue','royablue','steelblue']
        return colors


    #11 生成标记
    def getRandomMarker(self):
        markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
        #markers = ['s','o', '*']   
        random.shuffle(markers) 
        return markers
