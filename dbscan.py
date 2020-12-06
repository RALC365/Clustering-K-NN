import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

archivoAnalizar = 'datos_' + str(input("Ingrese el índice del archivo a analizar (1,2 o 3): ")) + '.csv'
eps_ = float(input("Ingrese el eps (0.25,0.35 o 0.5): "))
samples_ = int(input ("Ingrese el min_samples (5,10 o 15):"))

points = np.genfromtxt(archivoAnalizar, delimiter=',', skip_header=1)
plt.title(archivoAnalizar + '- eps =' + str(eps_) + ' min_samples = ' + str(samples_))

cluster = DBSCAN(eps=eps_, min_samples=samples_, metric='euclidean').fit(points)
print(cluster.labels_)

plt.scatter(points[:,0],points[:,1], c=cluster.labels_, cmap='cool')
plt.show()
