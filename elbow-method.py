from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH_SOURCE = '17_to_19_standardization_v3.csv'
selected_properties = ['TM', 'TMA', 'AverageTip', 'RST',
                       'RSI', 'RSE', 'Time', 'CorrectSteps', 'TriedSteps']
src_df = pd.read_csv(PATH_SOURCE, delimiter=',', usecols=selected_properties)
src_arr = src_df.to_numpy()

# Custom Elbow Method
distortions = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k).fit(src_arr)
    distortions.append(sum(np.min(cdist(src_arr, kmeans.cluster_centers_, 'euclidean'), axis=1)) / src_arr.shape[0])

plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# Yellowbrick Elbow Method
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1, 10))
visualizer.fit(src_arr)
visualizer.show()