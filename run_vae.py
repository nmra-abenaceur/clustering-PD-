# Torch and torchvision imports
import sys
import logging
from datetime import datetime
#exec_time = str(datetime.now())
logging.basicConfig(filename="files/logfile.log", level=logging.INFO)
#logging.basicConfig(filename="log_files/logfile_" + exec_time + ".log", level=logging.INFO)
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Basic data science imports
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sklearn scipy, statsmodels
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.cluster import KMeans, spectral_clustering
from sklearn.utils import resample

from statsmodels.stats.multitest import fdrcorrection

from math import sqrt
from statistics import median
from tqdm import tqdm
import pickle

from scipy.stats import pearsonr
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from scipy.stats import randint

from netneurotools import cluster
import netneurotools 

from snf import compute
from snf.cv import zrand_convolve

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import silhouette_score
from sklearn.exceptions import NotFittedError

from py_pcha import PCHA

from sklearn.manifold import TSNE

#Set Seeds
np.random.seed(614)
torch.manual_seed(614)

#---------------------------------------------------------------------
#VAE and Clustering Functions
# Variational Autoencoder Architecture
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = nn.functional.relu(self.fc1(x)) 
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h2 = nn.functional.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h2))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

#---------------------------------------------------------------------
# Loss function
def loss_function(recon_x, x, mu, logvar):
    reconst_loss = nn.functional.mse_loss(recon_x, x, reduction='sum') 
    KL_div_loss = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # D_KL( q_phi(z | x) || p_theta(z) )

    return reconst_loss, KL_div_loss

#---------------------------------------------------------------------
# Define training iteration function
def train(model, device, optimizer, kl_div_loss_weight = 0.1):
    """
    kl_div_loss_weight : float (optional)
        controls the balance between reconst_loss and KL div (higher means more weight
        on the KL_div part of the loss). This is Beta in the MVAE paper
    """
    model.train() 
    
    train_loss = 0
    running_reconst_loss = 0.0
    running_KL_div_loss = 0.0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        reconst_loss, KL_div_loss = loss_function(recon_batch, data, mu, logvar)
        loss = reconst_loss + kl_div_loss_weight * KL_div_loss
        loss.backward()

        # Update running loss totals 
        train_loss += loss.item()
        running_reconst_loss += reconst_loss.item()
        running_KL_div_loss += KL_div_loss.item()

        optimizer.step()

    total_loss = train_loss / len(train_loader.dataset) # divide by number of training samples
    reconst_loss = running_reconst_loss / len(train_loader.dataset)
    KL_div_loss =  running_KL_div_loss / len(train_loader.dataset)

    return total_loss, reconst_loss, KL_div_loss
#---------------------------------------------------------------------


# Define test iteration function
def test(model, device, kl_div_loss_weight = 0.1):
    model.eval() 
    test_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            reconst_loss, KL_div_loss = loss_function(recon_batch, data, mu, logvar)
            loss = reconst_loss + kl_div_loss_weight * KL_div_loss

            test_loss += loss.item()
    
    test_loss /= len(test_loader.dataset)

    return test_loss
#---------------------------------------------------------------------


#Spectral Clustering
def cluster_labels(embed):
    affinities = compute.make_affinity(embed, metric='cosine')
    #print("affinities = ", affinities)
    first, second = compute.get_n_clusters(affinities)
    fused_labels = spectral_clustering(affinities, n_clusters=first)#this is not deteministic
    #print('second : ', second)
    return fused_labels,first
#---------------------------------------------------------------------

#Consensus Clustering
def consenus(embed, n):
    
    ci=[]
    
    for i in range(n):
        fused_labels,first=cluster_labels(embed)#could be optimized to compute affinities and n_clusters only once
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

#Get Data
X1 = pd.read_csv("files/soma_matrix_for_VAE.csv", index_col=0)

disease_type = pd.read_csv("files/disease_type.csv", index_col=0)

# Min-Max Scale the data
scaler = MinMaxScaler()
X_scaled1 = scaler.fit_transform(X1)


# Get only the clinical-scale features (omitting those marked as Covariate, if applicable)
df1 = pd.DataFrame(X_scaled1, columns=X1.columns, index=X1.index)
scales_columns = df1.columns[ df1.columns.str.startswith( 'Protein' ) ].tolist()

df = df1[scales_columns]
X_scaled = df.values

n_participant, input_dim = X_scaled.shape
print("(n_patnos, input_dim) = (", n_participant, input_dim, ")")
#print(X1.columns)

X_train, X_val = train_test_split(X_scaled, train_size = 0.8, shuffle=True)
X_train_tensor, X_val_tensor = torch.Tensor(X_train), torch.Tensor(X_val)
BATCH_SIZE = 64

train_loader = DataLoader(
    X_train_tensor,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    X_val_tensor,
    batch_size=BATCH_SIZE,
    shuffle=False
)
#Initiate VAE
ii = 0
hyperparams = []
for line in open("tune_vae.txt"):
    txt_row = line.split()
    logging.info(str(txt_row))
    hyperparams.append(txt_row[1])
    ii = ii + 1

run_id = int(hyperparams[0])
latent_dim = int(hyperparams[1])
hidden_dim = int(hyperparams[2])
epochs = int(hyperparams[3])
learning_rate = 10**int(hyperparams[4])
kl_div_loss_weight = float(hyperparams[5])


#device = torch.device("cuda" )
device = torch.device("cpu" ) 
model =  VAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train VAE
train_loss_all_epochs = []
reconst_loss_all_epochs = []
KL_div_loss_all_epochs = []
test_loss_all_epochs = []

for epoch in range(0, epochs):
    total_loss, reconst_loss, KL_div_loss = train(model, device, optimizer, kl_div_loss_weight)
    test_loss = test(model, device, kl_div_loss_weight=kl_div_loss_weight)
    train_loss_all_epochs.append(total_loss)
    reconst_loss_all_epochs.append(reconst_loss)
    KL_div_loss_all_epochs.append(KL_div_loss)
    test_loss_all_epochs.append(test_loss)
    
# Plot loss
plt.figure()
sns.set(rc={'figure.figsize':(8,6)})
plt.plot(train_loss_all_epochs,label='Total Train Loss')
plt.plot(reconst_loss_all_epochs, label='Reconstruction Loss')
plt.plot(KL_div_loss_all_epochs, label='KLD Loss')
plt.plot(test_loss_all_epochs, label='Test Loss')
plt.title('Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend() 
plt.savefig('figs/02_train_test_loss.png')

# Create embedding for all participants
data1 = torch.Tensor(X_scaled).to(device)
mu, logvar = model.encode(data1)
embed = mu.cpu().detach().numpy()

projection = pd.DataFrame(embed, index=df.index)
#print(projection.shape)

#Visualizing the projection
plt.figure()
sns.set(rc={'figure.figsize':(10,15)})
ax=sns.heatmap(embed, cmap='bwr',center=0.00,)
ax.set(xlabel='Embedding', ylabel='Participant')
ax.set_title('Projection of participants in latent space')
plt.savefig('figs/02_proj_in_latent_space.png')


#Perform SVD
from numpy.linalg import svd
cols = X1.columns
U, sgv, V = svd(X1[cols].values)
#print('singular values', sgv)
cum_sgv = np.cumsum(sgv)/np.sum(sgv)
rank_80 = next(x for x, val in enumerate(cum_sgv) if val > 0.8)
rank_90 = next(x for x, val in enumerate(cum_sgv) if val > 0.9)
rank_95 = next(x for x, val in enumerate(cum_sgv) if val > 0.95)
nb_sgv = min(X1.shape)
plt.figure()
plt.plot(np.arange(nb_sgv)+1, sgv, color = 'b', linewidth = 1.5)
plt.yscale('log')
plt.xlabel('rank')
plt.ylabel('singular value')
plt.axvline(x = rank_80, color = 'y', label = '80% cumulative threshold', linewidth = 1.5)
plt.axvline(x = rank_90, color = 'r', label = '90% cumulative threshold', linewidth = 1.5)
plt.axvline(x = rank_95, color = 'g', label = '95% cumulative threshold', linewidth = 1.5)
plt.legend()
plt.savefig('figs/02_sgv_protein_set.png')



# R^2 on all participants (train and validation)
reconstruction = model.decode(mu)
r2_train_valid = r2_score(data1.cpu().detach().numpy(), reconstruction.cpu().detach().numpy(), multioutput='variance_weighted')
print("train & validation r2_score : "+str(r2_train_valid))
if r2_train_valid<0:
    print("Exiting script because an R2<0 error")
    sys.exit()
logging.info("test & validation r2_score : "+ str(r2_train_valid))

# R^2 on validation set
mu_val, _ = model.encode(X_val_tensor.to(device))
reconst_val = model.decode(mu_val)
r2_validation_variance_weighted = r2_score(X_val_tensor.cpu().detach().numpy(), reconst_val.cpu().detach().numpy(), multioutput='variance_weighted')
print("validation r2_score : "+str(r2_validation_variance_weighted))
logging.info("validation r2_score : "+ str(r2_validation_variance_weighted))

#X_val_tensor

fused_labels, clusters = consenus(embed, 30)
print('Number of clusters:',clusters)
logging.info('Number of clusters:'+ str(clusters))

cluster_stability = calculate_metrics(embed, clusters, fused_labels)
stability = median(cluster_stability)
print('---------------------------------Clustering Stability:', stability)
logging.info('Clustering Stability:'+ str(stability))

from math import log10

"""
nb_patients = X1.shape[0]
f = open("summary_nbp="+str(nb_patients)+".csv", "a")
f.write(str(run_id)+","+str(latent_dim)+","+str(hidden_dim)+","+str(epochs)+","+str(log10(learning_rate))+","+str(kl_div_loss_weight)+","+str(clusters)+","+str(stability)+"\n")
f.close()
"""

# Distribution of clusters
projection['group'] = fused_labels
projection['group'].value_counts()

#Visualization of clusters in 2D using TSNE


plt.figure()
sns.set(rc={'figure.figsize':(6,6)})

tsne = TSNE(n_components=2,n_jobs=-1)
tsne_results = tsne.fit_transform(embed)

ax=sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1], hue=projection['group'], palette="deep")

ax.set(xlabel='TSNE 1', ylabel='TSNE 2')
plt.savefig('figs/02_tsne.png')

# Look at by-feature R^2
r2_raw = r2_score(X_val_tensor.cpu().detach().numpy(), reconst_val.cpu().detach().numpy(), multioutput='raw_values')
r2_raw = pd.Series(r2_raw, index=df.columns)

with open('files/colnames_for_plots.json', 'r') as openfile:
    colnames_for_plots = json.load(openfile)

plt.figure()
r2_raw.rename(index=colnames_for_plots, inplace=True)
fig, ax = plt.subplots(figsize=(8,14))
r2_raw.sort_values()
r2_raw.sort_values(ascending=False)[0:100].plot.barh()
plt.title(f"Reconstruction R2 Score By Feature (validation set) \n Variance-weighted avg. = {r2_validation_variance_weighted:0.2f}", fontsize=18)
plt.gcf().savefig('figs/02_feature_reconstruction.png', bbox_inches='tight')

iter1={}
iter1['projection']=projection

#Projections on archetypes + Clusters
df_dbs=iter1['projection']
df_dbs= df_dbs.add_prefix('Projection_')
df_dbs= df_dbs.rename(columns={'Projection_group': 'group'})
#print("Projections and clusters :\n", df_dbs)

group_counts = df_dbs['group'].value_counts().sort_index(ascending=False).sort_index()#.plot.barh()
#sns.set_theme(style="whitegrid")
#sns.set(rc={'axes.facecolor':(0,0,0,0), 'figure.facecolor':(1,1,1,1)})
logging.info("group counts : \n" + str(group_counts))
#Vertical orientation
fig, ax = plt.subplots()
ax = sns.barplot(y=group_counts.values, x=group_counts.index,)
ax.bar_label(ax.containers[0],fontsize= 20)
ax.set_xticklabels([f'Cluster {i}' for i in group_counts.index], fontsize=14)
#ax.set_yticks([])
#ax.set_ylim([0,105])
#ax.set_ylabel("# participants", fontsize=20)
ax.set_title("Number of participants per cluster", fontsize=24)
sns.despine(bottom = True, left = True)
plt.xticks(rotation=45)

fig.savefig('figs/02_cluster_count_plot.png', dpi=300, bbox_inches='tight')

plt.figure()
sns.set(rc={'figure.figsize':(6,6)})
fig = plt.figure()
sns.set_palette('tab10')
sns.set_color_codes("muted")
ax=sns.scatterplot(x=embed[:,0], y=embed[:,1], hue=df_dbs['group'], palette='tab10')

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, prop={'size': 16})

plt.xlabel('UMAP 0', fontsize=24)
plt.ylabel('UMAP 1', fontsize=24)
plt.title('VAE Clusters', fontsize=24)
#ax.set_xticks([0,2,4,6,8],fontsize=18)
#ax.set_yticks([0,2,4,6,8],fontsize=18)
fig.savefig('figs/02_UMAP_clusters.png')#, dpi=1000, bbox_inches='tight')

df_cluster_type = disease_type.set_index('PATNO').join(pd.DataFrame(projection['group']))
df_cluster_type.to_csv('files/res/02_patno_type_cluster_runid='+str(run_id)+'.csv')
pd.DataFrame(embed).to_csv("files/res/02_embed_runid="+str(run_id)+".csv")

#-----------------------------------------------------------------------------
def closest_point(point, points):
    dist_2 = np.sum((points - point)**2, axis=1)
    return np.argmin(dist_2), np.min(dist_2)

#-----------------------------------------------------------------------------
def generate_archetypes(embed,noc,delta,df):
    
    XC, S, C, SSE, varexpl = PCHA(embed.transpose(), noc=noc, delta=delta)
    archetype_mat=XC.transpose()
    
    m1,n1=archetype_mat.shape
    
    archetype_point=[]
    for i in range(m1):
        point_num, point_dist=closest_point(archetype_mat[i].A1, embed)
        archetype_point.append(point_num)
        print(point_num, ", distance to closest archetype: ", point_dist)
    return df.iloc[archetype_point]
#-----------------------------------------------------------------------------

n_hidden = 5
archetype_list=generate_archetypes(embed,n_hidden,0.01,df).index.tolist()


archetype_load=df.loc[archetype_list]
plt.figure()
sns.set(rc={'figure.figsize':(25,5)})
ax=sns.heatmap(archetype_load, cmap='bwr', center=0.00)
ax.set(xlabel='Archetype', ylabel='Gene')
ax.set_title('Archetype Loading Matrix')
plt.savefig('figs/02_Archetype_Loading_Matrix.png')


#==================================================================================================================
import seaborn as sns
def compute_count_matrix(df_cluster_type, nb_clusters, stability):
    types = set(list(disease_type['DxGx'])) 
    nb_types = len(types)
    count_matrix = np.matrix(np.zeros((nb_clusters, nb_types)))
    for cl in range(nb_clusters):
        for idty, ty in enumerate(types):
            count_matrix[cl,idty] = len(set(df_cluster_type[df_cluster_type['group'] == cl+1].index) & set(df_cluster_type[df_cluster_type['DxGx'] == ty].index))
    plt.figure()
    sns.heatmap(count_matrix.T, square = True, annot=True, xticklabels = ["cluster "+str(ii) for ii in range(1,nb_clusters+1)], yticklabels = types, fmt = '.0f' )
    plt.title("cluster stability = "+str(stability))
    plt.margins(x=0.1, y=0.1)
    plt.savefig('figs/02_cluster_type.png', bbox_inches='tight')
    return count_matrix
#==================================================================================================================

count_matrix = compute_count_matrix(df_cluster_type, clusters, stability)

"""
#average protein expression
cols = X1.columns
centered_X1 = X1.copy(deep=True)
centered_X1[cols] = centered_X1[cols].subtract(centered_X1[cols].median(axis=0))#subtract average protein expression

clusterMedians = []
clusterProteins = []
column_names = []
for cl in range(clusters):
    cluster_patnos = list(df_cluster_type[df_cluster_type['group'] == cl+1].index)
    median_series = centered_X1[cols][centered_X1.index.isin(cluster_patnos)].median()
    median_argsort = abs(median_series).argsort()
    median_sort = median_series[median_argsort]
    clusterMedians.append(median_series)
    median_proteins = list(median_sort.index)
    median_proteins.reverse()
    median_expressions = list(median_sort.values)
    median_expressions.reverse()
    clusterProteins.append(median_proteins)
    clusterProteins.append(median_expressions)
    column_names.append('Cluster_'+str(cl+1))
    column_names.append('Cluster_'+str(cl+1))

#add median profile for all patients' datset
median_all = X1[cols].median()
median_argsort = abs(median_all).argsort()
median_sort = median_all[median_argsort]
clusterMedians.append(median_all)
median_proteins = list(median_sort.index)
median_proteins.reverse()
median_expressions = list(median_sort.values)
median_expressions.reverse()
clusterProteins.append(median_proteins)
clusterProteins.append(median_expressions)
column_names.append('all_patnos')
column_names.append('all_patnos')

cluster_profiles = pd.DataFrame(clusterProteins).T
cluster_profiles.columns = column_names
cluster_profiles.to_csv("files/02_cluster_proteins.csv")
"""

"""
nb_proteins = np.load("files/nb_proteins.npy")
#==================================================================================================================
patno_cluster_2 = pd.read_csv("files/02_patno_type_cluster_k=2.csv", index_col=0)
patno_cluster_5 = pd.read_csv("files/02_patno_type_cluster_k=5.csv", index_col=0)
def compute_confusion_matrix(patno_cluster_2, patno_cluster_5):
    confusion_matrix = np.matrix(np.zeros((5,2)))
    for cl2 in range(2):
        for cl5 in range(5):
            confusion_matrix[cl5,cl2] = len(set(patno_cluster_2[patno_cluster_2['group'] == cl2+1].index) & set(patno_cluster_5[patno_cluster_5['group'] == cl5+1].index))
    plt.figure()
    sns.heatmap(confusion_matrix.T, square = True, annot=True, xticklabels = ["cluster "+str(ii) for ii in range(1,5+1)], yticklabels = ["cluster "+str(ii) for ii in range(1,2+1)], fmt = '.0f' )
    plt.title('Cluster intersection (pat,prot) = ('+ str(nb_patients)+','+str(nb_proteins)+')')
    plt.margins(x=0.1, y=0.1)
    plt.savefig('figs/02_cluster_intersection.png', bbox_inches='tight')
    return confusion_matrix
#==================================================================================================================
confusion_matrix = compute_confusion_matrix(patno_cluster_2, patno_cluster_5)
"""
df_embed = pd.DataFrame(embed, columns = ['dim '+str(ii+1) for ii in range(latent_dim)])
df_embed.to_csv('files/protein_projection_latent_space.csv')

print(str(latent_dim)+","+str(hidden_dim)+","+str(epochs)+","+str(log10(learning_rate))+","+str(kl_div_loss_weight)+","+str(clusters)+"\n")