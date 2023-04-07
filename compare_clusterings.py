import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns


from statistics import median
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.cluster import spectral_clustering
#import netneurotools
from netneurotools import cluster
from snf import compute

np.random.seed(12)
random.seed(12)

#---------------------------------------------------------------------
#Spectral Clustering
def cluster_labels(embed, nCl):
    affinities = compute.make_affinity(embed, metric='cosine')
    if nCl == None:
        first, second = compute.get_n_clusters(affinities)
        nCl = first
    fused_labels = spectral_clustering(affinities, n_clusters=nCl)#this is not deteministic
    #print('second : ', second)
    return fused_labels, nCl
#---------------------------------------------------------------------
#Consensus Clustering
def consenus(embed, n, nCl):
    ci=[]
    for i in range(n):
        fused_labels, nCl = cluster_labels(embed, nCl)#could be optimized to compute affinities and n_clusters only once
        ci.append(list(fused_labels))
    consensus = cluster.find_consensus(np.column_stack(ci), seed=1234)
    a,=np.unique(consensus).shape
    return consensus,a
#---------------------------------------------------------------------

# Compute Clustering Stability
def ami_cluster_stability(data, true_labels, k, split= 0.20):
    """
    print("split ", split)
    print('data ', np.shape(data))
    print('labels ', np.shape(true_labels))
    print(true_labels)
    """
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

#==================================================================================================================
def compute_confusion_matrix(res, file, argmax_stab, argmax_nCl):
    patno_cluster_a = pd.read_csv("files/res/02_patno_type_cluster_runid="+str(argmax_stab)+".csv", index_col=0)
    patno_cluster_b = pd.read_csv("files/res/02_patno_type_cluster_runid="+str(argmax_nCl)+".csv", index_col=0)
    a = res['nb_clusters'].loc[argmax_stab]
    b = res['nb_clusters'].loc[argmax_nCl]
    print(a,b)
    confusion_matrix = np.matrix(np.zeros((a,b)))
    for cla in range(a):
        for clb in range(b):
            confusion_matrix[cla,clb] = len(set(patno_cluster_a[patno_cluster_a['group'] == cla+1].index) & set(patno_cluster_b[patno_cluster_b['group'] == clb+1].index))
    plt.figure()
    sns.heatmap(confusion_matrix.T, square = True, annot=True, xticklabels = ["cluster "+str(ii) for ii in range(1,a+1)], yticklabels = ["cluster "+str(ii) for ii in range(1,b+1)], fmt = '.0f' )
    plt.title('Cluster intersection (pat,prot) = ('+ str(nb_patients)+','+str(nb_proteins)+')')
    plt.margins(x=0.1, y=0.1)
    plt.savefig(file, bbox_inches='tight')
    return confusion_matrix

#==================================================================================================================
def confusion(res, file, patno_cluster_a , patno_cluster_b):
    a = len(set(patno_cluster_a['group'].values))
    b = len(set(patno_cluster_b['group'].values))
    print(a,b)
    confusion_matrix = np.matrix(np.zeros((a,b)))
    for cla in range(a):
        for clb in range(b):
            confusion_matrix[cla,clb] = len(set(patno_cluster_a[patno_cluster_a['group'] == cla+1].index) & set(patno_cluster_b[patno_cluster_b['group'] == clb+1].index))
    plt.figure()
    sns.heatmap(confusion_matrix.T, square = True, annot=True, xticklabels = ["cluster "+str(ii) for ii in range(1,a+1)], yticklabels = ["cluster "+str(ii) for ii in range(1,b+1)], fmt = '.0f' )
    plt.title('Cluster intersection (pat,prot) = ('+ str(nb_patients)+','+str(nb_proteins)+')')
    plt.margins(x=0.1, y=0.1)
    plt.savefig(file, bbox_inches='tight')
    return confusion_matrix
#==================================================================================================================


nb_patients = np.load("files/nb_patients.npy")
nb_proteins = np.load("files/nb_proteins.npy")

res = pd.read_csv("summary_nbp="+str(nb_patients)+".csv", index_col=0)
res2 = res[res['stability']>=.85]
res2 = res2.sort_values("stability", ascending = False)
argmax_stab = res2['stability'].idxmax()
argmax_nCl = res2['nb_clusters'].idxmax()
print("---------")
print(res.loc[[argmax_stab, argmax_nCl]])
print("---------")


file = 'figs/cluster_intersection.png'
confusion_matrix = compute_confusion_matrix(res, file, argmax_stab, argmax_nCl)

import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles

# define the sets A, B, and C
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}
C = {4, 6, 7, 8}

# create a Venn diagram
venn3([A, B, C], set_colors=('red', 'green', 'blue'), set_labels=('A', 'B', 'C'))

# add circles for each set
venn3_circles([A, B, C], linestyle='dashed')

# set title and show the diagram
plt.title('Intersection of Sets A, B, and C')
plt.show()

k = kk

# View confusion matrix between nCl = 2 and nCl = 5
nCl = res2['nb_clusters'].loc[argmax_nCl]
embed = pd.read_csv("files/res/02_embed_runid="+str(argmax_stab)+".csv")
embed_nCl = pd.read_csv("files/res/02_embed_runid="+str(argmax_nCl)+".csv")
patno_cluster_2c = pd.read_csv("files/res/02_patno_type_cluster_runid="+str(argmax_stab)+".csv")[['PATNO', 'group']]
fused_labels, _ = consenus(embed, 30, nCl)
print('Number of clusters:', nCl)

cluster_stability = calculate_metrics(embed, nCl, fused_labels)
stability = median(cluster_stability)

disease_type = pd.read_csv("files/disease_type.csv", index_col=0)
X1 = pd.read_csv("files/soma_matrix_for_VAE.csv", index_col=0)
projection = pd.DataFrame(embed, index=X1.index)
projection['group'] = fused_labels
df_cluster_type = disease_type.set_index('PATNO').join(pd.DataFrame(projection['group']))
df_cluster_type.to_csv("files/res/02_patno_type_cluster_runid=nested.csv")
print('---------------------------------Clustering Stability:', stability)


#Recluster the 5 clusters into 2:
#1 - Get embeddings for both clusters
fused_labels_5to2, nCl_5to2 = consenus(embed_nCl, 30, 2)
df_labels_5to2 = pd.DataFrame(fused_labels_5to2, columns = ['group'])
patno_cluster_5c = pd.read_csv("files/res/02_patno_type_cluster_runid="+str(argmax_nCl)+".csv")[['PATNO', 'group']]
df_labels_5c = pd.DataFrame(patno_cluster_5c['group'])
file = 'figs/cluster_intersection_5to2.png'
confusion_matrix = confusion(res, file, df_labels_5to2, df_labels_5c)

#Recluster from 2 to 5 clusters
fused_labels_2to5, nCl_2to5 = consenus(embed_nCl, 30, 5)
df_labels_2to5 = pd.DataFrame(fused_labels_2to5, columns = ['group'])
df_labels_2c = pd.DataFrame(patno_cluster_2c['group'])
file = 'figs/cluster_intersection_2to5.png'
confusion_matrix = confusion(res, file, df_labels_2to5, df_labels_2c)


#sub-cluster the 2 clusters:
#1 - Get embeddings for both clusters
embed_1 = embed[patno_cluster_2c['group']== 1]
embed_2 = embed[patno_cluster_2c['group']== 2]
#2 - Cluster using nCl = first
fused_labels_1stc, nCl_1 = consenus(embed_1, 30, None)
fused_labels_2ndc, nCl_2 = consenus(embed_2, 30, None)
fused_labels_2ndc = fused_labels_2ndc + nCl_2

df_labels_10 = pd.DataFrame(fused_labels_1stc, columns = ['group'], index = embed_1.index)
df_labels_20 = pd.DataFrame(fused_labels_2ndc, columns = ['group'], index = embed_2.index)
df_labels_2to6 =  pd.concat([df_labels_10, df_labels_20])

#df_labels_200 = pd.DataFrame(patno_cluster_2c['group'])
file = 'figs/cluster_intersection_2to6.png'
confusion_matrix = confusion(res, file, df_labels_2to6, df_labels_5to2)