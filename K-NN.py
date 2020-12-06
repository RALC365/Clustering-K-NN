#Libreria grafico de dispersión
import matplotlib.pyplot as plt
#Libreria para arreglo
import numpy as np
#Libreria para el K-NN
from sklearn.cluster import KMeans

#Extracción de puntos de archivo CSV
archivoAnalizar = 'datos_' + str(input("Ingrese el índice del archivo a analizar (1,2 o 3): ")) + '.csv'
k = int(input("K: Ingrese una cantidad entera de [1,5]: "))
points = np.genfromtxt(archivoAnalizar, delimiter=',', skip_header=1)

plt.title(archivoAnalizar + '- K=' + str(k))
plt.scatter(points[:,0],points[:,1], label='True Position')

kmeans = KMeans(n_clusters=k).fit(points)

print(kmeans.cluster_centers_)
print(kmeans.labels_)

#Pintando el gráfico resultante de dispersión
plt.scatter(points[:,0],points[:,1], c=kmeans.labels_, cmap='cool')

#Pintando los puntos centrales finales
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
plt.show()
   