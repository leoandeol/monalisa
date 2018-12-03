import matplotlib.pyplot as plt

from scipy import ndimage

from sklearn import cluster
import numpy as np
import math

image = ndimage.imread("monaLisa-100.jpg")

kmeans = cluster.KMeans(n_clusters=50)
x,y,z=image.shape
# a = np.ones((x,y))*range(-y//2,y//2)*2
# b = np.ones((y,x))*range(-x//2,x//2)*2
# b = b.T
# image[:,:,-1]=(a+b)
# print(a+b)
#image[:,:,-1]=b
#option 2
a = np.ndarray((x,y))
for i in range(x):
    for j in range(y):
        a[i,j]=math.sqrt(pow(x//2-i,2)+pow(y//2-j,2))*2
image[:,:,-1]=a

image2 = image.reshape(x*y,z)
r = kmeans.fit_predict(image2)
r2 = r.reshape(x,y)
print(r2)
plt.imshow(r2)
plt.show()
