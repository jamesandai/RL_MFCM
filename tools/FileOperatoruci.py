import numpy as np

class FileOperatoruci:

    def readWisc(self,filename):
        data = []
        label = []
        with open(filename, 'r') as f:
            while True:
                m = 0
                data_temp = []
                lines = f.readline().strip('\n')
                if not lines:
                    f.close()
                    break
                else:
                    for i in lines.split(','):
                        m = m + 1
                        if m == 10:
                            if i == 'malignant':
                                label.append(0)
                            else:
                                label.append(1)
                        else:
                            data_temp.append(float(i))
                data.append(data_temp)
        data = np.array(data)
        label = np.array(label)
        return data, label

    def readLymphography(self,filename):
        data = []
        label = []
        with open(filename, 'r') as f:
            while True:
                m = 0
                data_temp = []
                lines = f.readline().strip('\n')
                if not lines:
                    f.close()
                    break
                else:
                    for i in lines.split(','):
                        m = m + 1
                        if m == 1:
                            label.append(float(i))
                        else:
                            data_temp.append(float(i))
                data.append(data_temp)
        data = np.array(data)
        label = np.array(label)
        return data, label
    def readIris(self,filename):
        data = []
        label = []
        name = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        with open(filename, 'r') as f:
            while True:
                m = 0
                data_temp = []
                lines = f.readline().strip('\n')
                if not lines:
                    f.close()
                    break
                else:
                    for i in lines.split(','):
                        m = m + 1
                        if m == 5:
                            label.append(name.index(i))
                        else:
                            data_temp.append(float(i))
                data.append(data_temp)
        data = np.array(data)
        label = np.array(label)
        return data,label

    def readWine(self,filename):
        data = []
        label = []
        with open(filename,"r") as f:
            while True:
                m = 0
                data_temp = []
                lines = f.readline()
                if not lines:
                    break
                else:
                    for i in lines.split(','):
                        m = m + 1
                        if  m == 1:
                            label.append(float(i))
                        else:
                            data_temp.append(float(i))
                data.append(data_temp)
        data = np.array(data)
        label = np.array(label)
        return data,label

    def readSegmentation(self,filename):
        data = []
        label = []
        name = ['BRICKFACE', 'SKY', 'FOLIAGE', 'CEMENT', 'WINDOW', 'PATH', 'GRASS']
        with open(filename, 'r') as f:
            while True:
                m = 0
                data_temp = []
                lines = f.readline()
                if not lines:
                    f.close()
                    break
                else:
                    for i in lines.split(','):
                        m = m + 1
                        if m == 1:
                            label.append(name.index(i))
                        else:
                            data_temp.append(float(i))
                data.append(data_temp)
        data = np.array(data)
        label = np.array(label)
        return data,label

    def readDermetology(self,filename):
        data = []
        label = []
        miss_index = np.zeros(366)
        with open(filename, "r") as f:
            h = 0
            while True:
                lines = f.readline()
                data_temp = []
                if not lines:
                    break
                else:
                    lines = lines.strip('\n')
                    m = 0
                    if '?' in lines:
                        data_temp.append(-1)
                        miss_index[h] = 1
                    else:
                        for i in lines.split(','):
                            if m == 34:
                                label.append(float(i))
                            else:
                                data_temp.append(float(i))
                            m += 1
                        data.append(data_temp)
                h += 1
        data = np.array(data)
        label = np.array(label)
        mean = 0
        size, dim = data.shape
        for i in range(size):
            if miss_index[i] == 1:
                pass
            else:
                mean += data[i, 33]
        for i in range(size):
            if miss_index[i] == 1:
                data[i, 33] = mean / 358
        return data, label

    def readSeed(self,filename):
        data = []
        data_label = []
        with open(filename, "r") as f:
            while True:
                line = f.readline()
                m = 0
                data_tmp = []
                if not line:
                    break
                for i in line.split('	'):
                    if (m == 7):
                        data_label.append(int(i) - 1)
                    else:
                        data_tmp.append(float(i))
                    m = m + 1
                data.append(data_tmp)
        data = np.array(data)
        data_label = np.array(data_label)
        return data,data_label

    def readWdbc(self,filename):
        data = []
        data_label = []
        with open(filename, "r") as f:
            while True:
                data_tmp = []
                k = 0
                lines = f.readline()
                if not lines:
                    break
                lines = lines.strip('\n')
                for i in lines.split(","):
                    k = k + 1
                    if k == 1:
                        pass
                    elif k == 2:
                        if (i == "M"):
                            data_label.append(0)
                        else:
                            data_label.append(1)
                    else:
                        data_tmp.append(float(i))
                data.append(data_tmp)
        data = np.array(data)
        data_label = np.array(data_label)
        return data,data_label

    def readZoo(self,filename):
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
                for i in line.split(','):
                    if m == 0:
                        pass
                    elif m == 17:
                        label.append(int(i))
                    else:
                        data_tmp.append(float(i))
                    m = m + 1
                data.append(data_tmp)
        data = np.array(data)
        label = np.array(label)
        return data,label

    def readSonar(self,filename):
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
                for i in line.split(','):
                    if m == 60:
                        if i =='R':
                            label.append(1)
                        elif i == 'M':
                            label.append(2)
                    else:
                        data_tmp.append(float(i))
                    m = m + 1
                data.append(data_tmp)
        data = np.array(data)
        label = np.array(label)
        return data, label

    def readNewthyroid(self,filename):
        data = []
        label = []
        with open(filename, "r") as f:
            while True:
                m = 0
                data_temp = []
                lines = f.readline()
                if not lines:
                    break
                else:
                    for i in lines.split(','):
                        m = m + 1
                        if m == 1:
                            label.append(float(i))
                        else:
                            data_temp.append(float(i))
                data.append(data_temp)
        data = np.array(data)
        label = np.array(label)
        return data, label

    def readBupa(self,filename):
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
                for i in line.split(','):
                    if m == 6:
                        label.append(int(i))
                    else:
                        data_tmp.append(float(i))
                    m = m + 1
                data.append(data_tmp)
        data = np.array(data)
        label = np.array(label)
        return data, label

    def readHaberman(self,filename):
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
                for i in line.split(','):
                    if m == 3:
                        label.append(int(i))
                    else:
                        data_tmp.append(float(i))
                    m = m + 1
                data.append(data_tmp)
        data = np.array(data)
        label = np.array(label)
        return data, label

    def readControl(self,filename):
        data = []
        label = []
        with open(filename, "r") as f:
            k = 0
            while True:
                line = f.readline()
                data_tmp = []
                line = line.strip('\n')
                if not line:
                    break
                for i in line.split(' '):
                    print(i)
                    data_tmp.append(float(i))
                label.append(int(k / 100))
                k += 1
                data.append(data_tmp)
        data = np.array(data)
        label = np.array(label)
        return data, label

    def readDigits(self,filename):
        data = []
        label = []
        with open(filename, 'r') as f:
            while True:
                m = 0
                data_temp = []
                lines = f.readline().strip('\n')
                if not lines:
                    f.close()
                    break
                else:
                    for i in lines.split(','):
                        m = m + 1
                        if m == 65:
                            label.append(float(i))
                        else:
                            data_temp.append(float(i))
                data.append(data_temp)
        data = np.array(data)
        label = np.array(label)
        return data, label

    def readBanknote(self,filename):
        data = []
        label = []
        with open(filename, 'r') as f:
            while True:
                m = 0
                data_temp = []
                lines = f.readline().strip('\n')
                if not lines:
                    f.close()
                    break
                else:
                    for i in lines.split(','):
                        m = m + 1
                        if m == 5:
                            label.append(float(i))
                        else:
                            data_temp.append(float(i))
                data.append(data_temp)
        data = np.array(data)
        label = np.array(label)
        return data, label

    def readBalance(self,filename):
        data = []
        label = []
        with open(filename, 'r') as f:
            while True:
                m = 0
                data_temp = []
                lines = f.readline().strip('\n')
                if not lines:
                    f.close()
                    break
                else:
                    for i in lines.split(','):
                        m = m + 1
                        if m == 1:
                            if i == 'B':
                                label.append(0)
                            elif i == 'R':
                                label.append(1)
                            else:
                                label.append(2)
                        else:
                            data_temp.append(float(i))
                data.append(data_temp)
        data = np.array(data)
        label = np.array(label)
        return data, label

    def readParkinsons(self,filename):
        data = []
        label = []
        with open(filename, 'r') as f:
            while True:
                m = 0
                data_temp = []
                lines = f.readline().strip('\n')
                if not lines:
                    f.close()
                    break
                else:
                    for i in lines.split(','):
                        m = m + 1
                        if m == 1:
                            pass
                        elif m == 18:
                            label.append(float(i))
                        else:
                            data_temp.append(float(i))
                data.append(data_temp)
        data = np.array(data)
        label = np.array(label)
        return data, label

    def readEcoli(self,filename):
        data = []
        label = []
        with open(filename, 'r') as f:
            while True:
                m = 0
                data_temp = []
                lines = f.readline().strip('\n')
                if not lines:
                    f.close()
                    break
                else:
                    for i in lines.split('  '):
                        m = m + 1
                        if m == 1:
                            pass
                        elif m == 9:
                            if i == ' cp':
                                label.append(1)
                            elif i == ' im':
                                label.append(2)
                            elif i == 'imS':
                                label.append(3)
                            elif i == 'imL':
                                label.append(4)
                            elif i == 'imU':
                                label.append(5)
                            elif i == ' om':
                                label.append(6)
                            elif i == 'omL':
                                label.append(7)
                            elif i == ' pp':
                                label.append(8)
                        else:
                            data_temp.append(float(i))
                data.append(data_temp)
        data = np.array(data)
        label = np.array(label)
        return data, label

    def readCMC(self,filename):
        data = []
        label = []
        with open(filename, 'r') as f:
            while True:
                m = 0
                data_temp = []
                lines = f.readline().strip('\n')
                if not lines:
                    f.close()
                    break
                else:
                    for i in lines.split(','):
                        m = m + 1
                        if m == 10:
                            label.append(float(i))
                        else:
                            data_temp.append(float(i))
                data.append(data_temp)
        data = np.array(data)
        label = np.array(label)
        return data, label

    def readDRD(self,filename):
        data = []
        label = []
        with open(filename, 'r') as f:
            while True:
                m = 0
                data_temp = []
                lines = f.readline().strip('\n')
                if not lines:
                    f.close()
                    break
                else:
                    for i in lines.split(','):
                        m = m + 1
                        if m == 20:
                            label.append(float(i))
                        else:
                            data_temp.append(float(i))
                data.append(data_temp)
        data = np.array(data)
        label = np.array(label)
        return data, label

    def readWifi(self,filename):
        data = []
        label = []
        with open(filename, 'r') as f:
            while True:
                m = 0
                data_temp = []
                lines = f.readline().strip('\n')
                if not lines:
                    f.close()
                    break
                else:
                    for i in lines.split('\t'):
                        m = m + 1
                        if m == 8:
                            label.append(float(i))
                        else:
                            data_temp.append(float(i))
                data.append(data_temp)
        data = np.array(data)
        label = np.array(label)
        return data, label

    def readBreast(self, filename):
        data = []
        label = []
        with open(filename, "r") as f:
            h = 0
            while True:
                lines = f.readline()
                data_temp = []
                if not lines:
                    break
                else:
                    lines = lines.strip('\n')
                    m = 0
                    for i in lines.split(','):
                        m += 1
                        if m == 1:
                            pass
                        elif m == 11:
                            label.append(i)
                        elif m == 7:
                            pass
                        else:
                            data_temp.append(float(i))
                    data.append(data_temp)
                    h += 1
        data = np.array(data)
        label = np.array(label)
        return data, label

    def readDiagnorse(self, filename):
        data = []
        label = []
        with open(filename, "r") as f:
            h = 0
            while True:
                lines = f.readline()
                data_temp = []
                if not lines:
                    break
                else:
                    lines = lines.strip('\n')
                    m = 0
                    for i in lines.split(','):
                        m += 1
                        if m == 10:
                            if  i == 'N':
                                label.append(0)
                            else:
                                label.append(1)
                        else:
                            data_temp.append(float(i))
                    data.append(data_temp)
                    h += 1
        data = np.array(data)
        label = np.array(label)
        size, dim = data.shape
        print(size,dim)
        return data, label


    def readGlass(self,filename):
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
                for i in line.split(','):
                    m = m + 1
                    if m == 1:
                        pass
                    elif m == 11:
                        label.append(int(i))
                    else:
                        data_tmp.append(float(i))
                data.append(data_tmp)
        data = np.array(data)
        label = np.array(label)
        return data,label

    def readColoumn(self,filename):
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
                for i in line.split(','):
                    m = m + 1
                    if m == 7:
                        if i == 'Hernia':
                            label.append(1)
                        elif i == 'Spondylolisthesis':
                            label.append(2)
                        elif i == 'Normal':
                            label.append(3)
                    else:
                        data_tmp.append(float(i))
                data.append(data_tmp)
        data = np.array(data)
        label = np.array(label)
        return data,label



    def readYeast(self,filename):
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
                for i in line.split('  '):
                    m = m + 1
                    if m == 1:
                        pass
                    elif m == 10:
                        if i == 'MIT':
                            label.append(1)
                        elif i == 'NUC':
                            label.append(2)
                        elif i == 'ME3':
                            label.append(3)
                        elif i == 'CYT':
                            label.append(4)
                        elif i == 'ME2':
                            label.append(5)
                        elif i == 'ME1':
                            label.append(6)
                        elif i == 'EXC':
                            label.append(7)
                        elif i == 'VAC':
                            label.append(8)
                        elif i == 'POX':
                            label.append(9)
                        else:
                            label.append(10)
                    else:
                        data_tmp.append(float(i))
                data.append(data_tmp)
        data = np.array(data)
        label = np.array(label)
        return data,label

    def readPage(self,filename):
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
                for i in line.split(','):
                    m = m + 1
                    if m == 11:
                        label.append(float(i))
                    else:
                        data_tmp.append(float(i))
                data.append(data_tmp)
        data = np.array(data)
        label = np.array(label)
        return data,label

    def readLym(self,filename):#还没测试
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
                for i in line.split(','):
                    m = m + 1
                    if m == 19:
                        label.append(float(i))
                    else:
                        data_tmp.append(float(i))
                data.append(data_tmp)
        data = np.array(data)
        label = np.array(label)
        return data,label

    def readAba(self,filename):#还没测试
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
                for i in line.split(','):
                    m = m + 1
                    if m == 1:
                        if i == 'M':
                            data_tmp.append(0)
                        elif i == 'I':
                            data_tmp.append(1)
                        else:
                            data_tmp.append(2)
                    elif m == 9:
                        label.append(float(i))
                    else:
                        data_tmp.append(float(i))
                data.append(data_tmp)
        data = np.array(data)
        label = np.array(label)
        return data,label

    def readDivorce(self,filename):#还没测试
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
                for i in line.split(';'):
                    m = m + 1
                    if m == 55:
                        label.append(float(i))
                    else:
                        data_tmp.append(float(i))
                data.append(data_tmp)
        data = np.array(data)
        label = np.array(label)
        return data,label