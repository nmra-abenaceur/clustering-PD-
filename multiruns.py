import os
import time
import random
import itertools
import numpy as np

start = time.time()
nb_samples = 400

random.seed(614)

nb_patients = np.load("files/nb_patients.npy")
nb_proteins = np.load("files/nb_proteins.npy")

f = open("summary_nbp="+str(nb_patients)+".csv", "w")
f.write("index,latent_dim,hidden_dim,epochs,learning_rate,kl_div_loss_weight,nb_clusters,stability\n")
f.close()

list_epochs = [90,100,110,120,130,140,150]
list_latent_dim = [2,3,4,5,6,7,8,9,10]
list_hidden_dim = [20,30,40,50,60]
list_learning_rate = [-2,-3]
list_kl_div_loss_weight = [.1, .5, 1]

somelists = [list_epochs,
             list_latent_dim,
             list_hidden_dim,
             list_learning_rate,
             list_kl_div_loss_weight]

cartesian_prod = list(itertools.product(*somelists))

rand_hyperpar_sample = random.sample(cartesian_prod, nb_samples)

run_id = 0
for epochs, latent_dim, hidden_dim, learning_rate, kl_div_loss_weight in rand_hyperpar_sample:
    f = open("tune_vae.txt", "w")
    f.write("index  "+str(run_id)+"\n")
    f.write("latent_dim  "+str(latent_dim)+"\n")
    f.write("hidden_dim  "+str(hidden_dim)+"\n")
    f.write("epochs  "+str(epochs)+"\n")
    f.write("learning_rate   "+str(learning_rate)+"\n")
    f.write("kl_div_loss_weight  "+str(kl_div_loss_weight)+"\n")
    f.close()
    os.system("python run_vae.py")
    run_id = run_id+1
    
total_time = time.time()-start
print('total time is ', total_time, ' s')
