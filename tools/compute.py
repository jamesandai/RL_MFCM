import numpy as np
A = [[0.41,0.29,0.3,0],[0,0.9,0.1,0],[0.58,0.07,0,0.35],[0.55,0.1,0,0.35],[0.6,0.1,0,0.3]]
D = [[1,0,0,0.5],[0,1,0,0],[0,0,1,0.5],[0.5,0,0.5,1]]
A = np.array(A)
D = np.array(D)
sum = 0
for i in range(1,5):
    k = np.sqrt((A[0] - A[i]).dot(D).dot((A[0]-A[i]).T) / 2)
    sum += k
    print(sum)
