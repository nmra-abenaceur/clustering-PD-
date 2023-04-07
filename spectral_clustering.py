import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

from snf import compute
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from statistics import median
from sklearn.cluster import spectral_clustering

from netneurotools import cluster

exec_time = str(datetime.now())
#---------------------------------------------------------------------
#Spectral Clustering
def cluster_labels(embed):
    affinities = compute.make_affinity(embed, metric='cosine')
    #print("affinities = ", affinities)
    first, second = compute.get_n_clusters(affinities)
    fused_labels = spectral_clustering(affinities, n_clusters=first)
    print("nb clusters ", first, second)
    return fused_labels,first
#---------------------------------------------------------------------
#Consensus Clustering
def consenus(embed, n):
    ci=[]
    for i in range(n):
        fused_labels,first=cluster_labels(embed)
        ci.append(list(fused_labels))   
    consensus = cluster.find_consensus(np.column_stack(ci), seed=1234)
    a,=np.unique(consensus).shape
    return consensus,a
#---------------------------------------------------------------------
# Compute Clustering Stability
def ami_cluster_stability(data, true_labels, k, split= 0.20):
    X_sample, X_rest, y_sample, y_rest = train_test_split(data, true_labels, test_size=split)
    affinities = compute.make_affinity(X_sample, metric='cosine')
    y_cluster = spectral_clustering(affinities, n_clusters=k)
    return adjusted_mutual_info_score(y_cluster, y_sample)
#---------------------------------------------------------------------

def calculate_metrics(data, n_clusters, labels, iterations=20):
    cluster_stability = []

    for i in range(iterations):
        cluster_stability.append(ami_cluster_stability(data, labels, n_clusters))
    
    return cluster_stability
#---------------------------------------------------------------------

X1 = pd.read_csv("files/soma_matrix_for_VAE.csv", index_col=0)


scaler = MinMaxScaler()
df1 = scaler.fit_transform(X1)


# Get only the clinical-scale features (omitting those marked as Covariate, if applicable)
df1 = pd.DataFrame(df1, columns=X1.columns, index=X1.index)

disease_type = pd.read_csv("files/disease_type.csv", index_col=0).set_index('PATNO')

fused_labels, nb_clusters = cluster_labels(df1)

cluster_stability = calculate_metrics(df1, nb_clusters, fused_labels)
stability = median(cluster_stability)
print('---------------------------------Clustering Stability:', stability)

data = {'PATNO': list(disease_type.index),
        'group': list(fused_labels+1)}
cluster_no = pd.DataFrame(data)
df_cluster_type = disease_type.join(cluster_no.set_index("PATNO"))

types = set(list(disease_type['DxGx'])) 
nb_types = len(types)
count_matrix = np.matrix(np.zeros((nb_clusters, nb_types)))
mean = []
df1mean = df1.mean(axis=0)
for cl in range(nb_clusters):
    mean.append((df1[df_cluster_type['group'] == cl+1].mean(axis=0)).values)
    for idty, ty in enumerate(types):
        count_matrix[cl,idty] = len(set(df_cluster_type[df_cluster_type['group'] == cl+1].index) & set(df_cluster_type[df_cluster_type['DxGx'] == ty].index))
plt.figure()
sns.heatmap(count_matrix.T, square = True, annot=True, xticklabels = ["cluster "+str(ii) for ii in range(1,nb_clusters+1)], yticklabels = types, fmt = '.0f' )
plt.margins(x=0.1, y=0.1)
plt.title("spectral clustering")
plt.savefig('figs/cluster_type.png', bbox_inches='tight')

mean_data = {'cluster'+str(ii): list(mean[ii]) for ii in range(nb_clusters)}
cluster_means = pd.DataFrame(mean_data)
cluster_means.index = X1.columns

plt.figure()
ax=sns.heatmap(cluster_means)
plt.savefig('figs/cluster_means.png', bbox_inches='tight')

"""fused_labels, clusters = consenus(df1, 30)
print('Number of clusters:',clusters)"""
