import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import Animation
import sys
import os
from cell import Cell
from graph import Graph
import math
import numpy as np

img = plt.imread('mask/t002mask.tif')
size = img.shape
width = size[0]
height  = size[1]

graph = Graph(img,height,width)

for  i in range(width):
    for j in range(height):
        if img[i][j] == 0:
            continue
        graph.addCell(img[i][j],i,j)
    
c1 = graph.get_cell(3)

e1 = c1.Ellipse()
c2 = graph.get_cell(6)

e2 = c2.Ellipse()


# c2.show(e2)

def merge(e1,e2):
    result = e2.copy()
    for i in range(e1.shape[0]):
        for j in range(e1.shape[1]):
            if e1[i][j] !=0:
                result[i][j] = e1[i][j]

    return result

# e3 = merge(e1,e2)

# c2.show(e3)

img = plt.imread('mask/t005mask.tif')
size = img.shape
width = size[0]
height  = size[1]

graph1 = Graph(img,height,width)

for  i in range(width):
    for j in range(height):
        if img[i][j] == 0:
            continue
        graph1.addCell(img[i][j],i,j)
    

img1 = plt.imread('mask/t021mask.tif')
size = img1.shape
width = size[0]
height  = size[1]

graph2 = Graph(img1,height,width)

for  i in range(width):
    for j in range(height):
        if img[i][j] == 0:
            continue
        graph2.addCell(img[i][j],j,i)
    

l = []
l.append(graph)
l.append(graph1)
l.append(graph2)

# for f in l:
#     f.showM()


def abs_v(value):
    x = value[0]
    y =value[1]
    v = math.sqrt(x**2+y**2)
    return v
def vector(x,y):
    x1 = x[0]-y[0]
    y1 = x[1]-y[1]

    return (x1,y1)
def displacement(cc1,cc2,height,width):
    x = cc1[0] - cc2[0]
    y = cc1[1] - cc2[1]
    v = abs_v((x,y))
    gv = abs_v((height,width))
    return v/gv

# def skewness(cc1,cc2):
#     # v1 = vector(cc1,cc2)
#     # v2 = vector(cc2,cc3)
#     # v3 = vector(v1,v2)

#     # return 0.5*(1 - abs_v(v3)/((abs_v(v1))*(abs_v(v2))))



def color(i1,i2):
    n_mean = np.mean(i2)
    sum = 0.0
    for i in i1:
        sum = sum + ((i-n_mean)/255)**2

    return np.sqrt(sum/len(i1))

def area(i1,i2):
    count = 0
    for i in range(i1.shape[0]):
        for j  in range(i1.shape[1]):
            if i1[i][j] !=0:
                count = count + 1
    
    s1 = count

    count = 0
    for i in range(i2.shape[0]):
        for j  in range(i2.shape[1]):
            if i1[i][j] !=0:
                count = count + 1
    
    s2 = count    

    count = 0
    for i in range(i1.shape[0]):
        for j  in range(i1.shape[1]):
            if i1[i][j] !=0:
                if i2[i][j]!=0:
                    count = count +1
    
    s3 = count

  

    return 1 - s3**2/(s1*s2)

def deformation(q1,q2):
    a = abs(q1-q2)
    b = np.sqrt(q1**2 + q2**2)

    return a/b

def Intensity(real,mask):
    l = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] !=0:
                l.append(real[i][j])
    return l


result = np.zeros((10,10))
real = plt.imread("dataset/DIC-C2DH-HeLa/Sequence 1/t002.tif")
real1 = plt.imread("dataset/DIC-C2DH-HeLa/Sequence 1/t005.tif")
# l = Intensity(real,e1)
# # print(l)
# graph.showM()
# graph1.showM()


for i in range(10):
    for j in range(10):

        cell1 = graph.get_cell(i+1)
        cell2 = graph1.get_cell(j+1)
        e1 = cell1.Ellipse()
        e2 = cell2.Ellipse()
        # cell1.show(e1)
        E1 = displacement(cell1.COM1(),cell2.COM1(),height,width)

        if i==4 and j ==0:
            print(1)

        E3 = color(Intensity(real,e1),Intensity(real1,e2))
        E4 = area(e1,e2)
        E5 = deformation(cell1.ecc,cell2.ecc)

        v = 0.3*E1  + 0.1* E3 + 0.3 *E4 +0.3 *E5
        result[i][j] = v

        print(f"{E1},{E3},{E4},{E5}")
        print(f"Current:{cell1.id}->{cell2.id}")

for i in range(10):
    
    match = np.argmin(result[i])
    print(f"find:{i}->{match}")


