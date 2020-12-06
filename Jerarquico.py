import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering

archivoAnalizar = 'datos_' + str(input("Ingrese el índice del archivo a analizar (1,2 o 3): ")) + '.csv'
decision = int(input("Ingrese 1 si desea clasificar por n_clusters y 2 si por distance_threshold: "))
points = np.genfromtxt(archivoAnalizar, delimiter=',', skip_header=1)
cluster = AgglomerativeClustering()

if(decision == 1):
    k = int(input("Ingrese la cantidad de k (1-5): "))
    plt.title(archivoAnalizar + '- K=' + str(k))
    cluster = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='single')
else:
    dt = float(input('Ingrese la distancia (0.25, 0.5, 0.75, 1.0, 1.5): '))
    plt.title(archivoAnalizar + '- UD=' + str(dt))
    cluster = AgglomerativeClustering(n_clusters=None ,affinity='euclidean',compute_full_tree=True ,linkage='single', distance_threshold= 1.5)


cluster.fit_predict(points)
print(cluster.labels_)

plt.scatter(points[:,0],points[:,1], c=cluster.labels_, cmap='cool')
plt.show()
