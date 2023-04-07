import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

nb_patients = np.load("files/nb_patients.npy")
nb_proteins = np.load("files/nb_proteins.npy")

summary = pd.read_csv("summary_nbp="+str(nb_patients)+".csv")
summary_sort = summary.sort_values("stability", ascending = False)
summary_sort.to_csv("sort_summary_nbp="+str(nb_patients)+".csv")

cl_stab_df = summary_sort[['nb_clusters', 'stability']]
clusters = list(set(cl_stab_df['nb_clusters']))
cluster_stab_vals = []
counts = []
counts_80 = []
counts_85 = []
counts_90 = []
for ii, cl in enumerate(clusters):
    df_cl = cl_stab_df[cl_stab_df['nb_clusters']==cl]
    counts.append(len(df_cl))
    cluster_stab_vals.append(df_cl['stability'].values)
    df_cl_80 = df_cl[df_cl['stability']>=.8]
    counts_80.append(len(df_cl_80))
    df_cl_85 = df_cl[df_cl['stability']>=.85]
    counts_85.append(len(df_cl_85))
    df_cl_90 = df_cl[df_cl['stability']>=.9]
    counts_90.append(len(df_cl_90))


fig, ax = plt.subplots()
ax = sns.barplot(y=counts, x=clusters, color = 'c', label='all')
ax = sns.barplot(y=counts_80, x=clusters, color = 'r', label = '>.80')
ax = sns.barplot(y=counts_85, x=clusters, color = 'm', label = '>.85')
ax = sns.barplot(y=counts_90, x=clusters, color = 'g', label = '>.90')
ax.bar_label(ax.containers[0],fontsize= 14)
ax.set_xticklabels(clusters, fontsize=14)
ax.set_title('Clustering occurrence (pat,prot) = ('+ str(nb_patients)+','+str(nb_proteins)+')')
ax.set_xlabel('number of clusters')
ax.set_ylabel('count')
ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
sns.despine(bottom = True, left = True)
fig.savefig('figs/06_cluster_occurrence.png', dpi=300, bbox_inches='tight')

plt.figure()
for ii, cl in enumerate(clusters):
    plt.plot(cluster_stab_vals[ii], label = str(cl))

plt.title('Stability per number of clusters (pat,prot) = ('+ str(nb_patients)+','+str(nb_proteins)+')')
plt.xlabel('occurrences')
plt.ylabel('stability')
plt.legend()
plt.savefig('figs/06_stability_nbcl.png', dpi=300, bbox_inches='tight')