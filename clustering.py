# The Effectiveness of Lockdowns, Face Masks and Vaccination Programmes Vis-à-Vis Mitigating COVID-19
# Martin Sewell
# martin.sewell@cantab.net
# 5 August 2024

######################
# K-MEANS CLUSTERING #
######################

import numpy as np
import os
from numpy import genfromtxt
from sklearn.cluster import KMeans

def smooth28(data):
    window = np.ones(int(28))/float(28)
    return np.convolve(data, window, mode='same')

c = open("clusters.txt","w", encoding='ascii')

Xcd = genfromtxt('clusteringcd.txt', delimiter='\t')
Xcd[np.isnan(Xcd)] = 0
Xcd = np.apply_along_axis(smooth28, 1, Xcd)
for num_clusters in range(2,21): # 2 to 20 inclusive (19 lines)
    modelcd = KMeans(n_clusters = num_clusters)
    labelscd = modelcd.fit_predict(Xcd)
    for i in labelscd:
        c.write(str(i)+'\t')
    c.write('\n')

Xem = genfromtxt('clusteringem.txt', delimiter='\t')
Xem[np.isnan(Xem)] = 0
Xem = np.apply_along_axis(smooth28, 1, Xem)
for num_clusters in range(2,21): # 2 to 20 inclusive (19 lines)
    modelem = KMeans(n_clusters = num_clusters)
    labelsem = modelem.fit_predict(Xem)
    for i in labelsem:
        c.write(str(i)+'\t')
    c.write('\n')

# 38 lines total

c.flush()
os.fsync(c.fileno())
c.close()
print("Finished!")