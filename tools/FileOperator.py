import numpy as np
 
class FileOperator:
    def readFourclass(self,fileName):
        data = []
        label = []
        with open(fileName, "r") as f:
            while True:
                line = f.readline()
                m = 0
                data_tmp = []
                line = line.strip('\n')
                if not line:
                    break
                for i in line.split(','):
                    if m == 3:
                        label.append(float(i))
                    elif m > 0 and m < 3:
                        data_tmp.append(float(i))
                    m = m + 1
                data.append(data_tmp)
        data = np.array(data)
        return data, label
    def readSpiral(self,fileName):
        data = []
        label = []
        with open(fileName, "r") as f:
            while True:
                line = f.readline()
                m = 0
                data_tmp = []
                line = line.strip('\n')
                if not line:
                    break
                for i in line.split(' '):
                    if m == 0:
                        label.append(float(i))
                    else:
                        data_tmp.append(float(i))
                    m = m + 1
                data.append(data_tmp)
        data = np.array(data)
        return data,label
    #1.从csv或者txt文件里获取无标签的数据点
    def readDatawithoutLabel(self, fileName):        
        points=[]   
        for line in open(fileName, "r"):
            items = line.strip("\n").split("\t")
            tmp = []
            for item in items:
                tmp.append(float(item))
            points.append(tmp)
        points = np.array(points)
        return points
    def readDatawithLabel_point(self, fileName):
        points = []
        label = []
        for line in open(fileName, "r"):
            items = line.strip("\n").split(" ")
            tmp = []
            i = 0
            for item in items:
                i += 1
                if i == 3:
                    label.append(float(item))
                else:
                    tmp.append(float(item))
            points.append(tmp)
        points = np.array(points)
        label = np.array(label)
        return points,label

    def readDatawithoutLabel_t8(self, fileName):
        points = []
        for line in open(fileName, "r"):
            items = line.strip("\n").split(" ")
            tmp = []
            for item in items:
                tmp.append(float(item))
            points.append(tmp)
        points = np.array(points)
        return points
    #2 从csv或者txt文件里获取有标签的数据点
    def readDatawithLabel(self, fileName):
        points=[]   
        labels = []
        for line in open(fileName, "r"):
            items = line.strip("\n").split(",")    #data format in each row:"x, y, label"
            labels.append(int(items.pop()))        #extract value of label
            tmp = []
            for item in items:
                tmp.append(float(item))
            points.append(tmp)                     #extract values of x and y
        points = np.array(points)
        labels = np.array(labels)
        return points,labels

    def readData3D(self,fileName):
        points = []
        labels = []
        for line in open(fileName, "r"):
            items = line.strip("\n").split(",")  # data format in each row:"x, y, label"
            tmp = []
            i = 0
            for item in items:
                i += 1
                if i == 4:
                    labels.append(float(item))
                else:
                    tmp.append(float(item))
            points.append(tmp)  # extract values of x and y
        points = np.array(points)
        labels = np.array(labels)
        return points,labels

    def readButterfly(self, fileName):
        points=[]
        labels = []
        for line in open(fileName, "r"):
            items = line.strip("\n").split(",")    #data format in each row:"x, y, label"
            tmp = []
            for item in items:
                tmp.append(float(item))
            points.append(tmp)                     #extract values of x and y
        points = np.array(points)
        return points

    def readDatawithLabel2(self, fileName):
        points=[]
        labels = []
        for line in open(fileName, "r"):
            items = line.strip("\n").split("\t")    #data format in each row:"x, y, label"
            labels.append(int(items.pop()))        #extract value of label
            tmp = []
            for item in items:
                tmp.append(float(item))
            points.append(tmp)                     #extract values of x and y
        points = np.array(points)
        labels = np.array(labels)
        return points,labels

    def readPoint(self, fileName):
        points=[]
        labels = []
        for line in open(fileName, "r"):
            items = line.strip("\n").split(" ")    #data format in each row:"x, y, label"
            tmp = []
            for item in items:
                tmp.append(float(item))
            k = tmp.pop()
            points.append(tmp)
            labels.append(k)
        points = np.array(points)
        labels = np.array(labels)
        return points,labels

    def readS1(self, fileName):
        points=[]
        for line in open(fileName, "r"):
            items = line.strip("\n").split("    ")    #data format in each row:"x, y, label"
            tmp = []
            i = 0
            for item in items:
                i += 1
                if i != 1:
                    tmp.append(float(item))
            points.append(tmp)
        points = np.array(points)
        return points

    def readSpiral(self,filename):
        data = []
        label = []
        with open(filename, "r") as f:
            while True:
                line = f.readline()
                m = 0
                data_tmp = []
                line = line.strip('\n')
                if not line:
                    break
                for i in line.split(' '):
                    if m == 0:
                        label.append(float(i))
                    else:
                        data_tmp.append(float(i))
                    m = m + 1
                data.append(data_tmp)
        data = np.array(data)
        return data,label

    def readThreeCircle(self,filename):
        data = []
        label = []
        with open(filename, "r") as f:
            while True:
                line = f.readline()
                m = 0
                data_tmp = []
                line = line.strip('\n')
                if not line:
                    break
                for i in line.split(' '):
                    if m == 0:
                        label.append(float(i))
                    else:
                        data_tmp.append(float(i))
                    m = m + 1
                data.append(data_tmp)
        data = np.array(data)
        return data,label
    
    
    #3 把数据存进文件里
    def writeData(self, data, fileName):
        f = open(fileName,'a')    #把数据添加在文件后面                                #内容之后写入。可修改该模式（'w+','w','wb'等）
        for d in data:
            f.write(str(d))   
            f.write("\n")       
        f.close()       
                