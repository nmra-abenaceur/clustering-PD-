import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#---------------------------------------------------------------------------------
def protein_std(datafile, cols):
    stds = []
    for cc in cols:
        stds.append(np.std(datafile[cc].values))
    std_df = pd.DataFrame(stds, index = cols, columns = ['standard_deviation'])
    return std_df
#---------------------------------------------------------------------------------

data_path = "/shared-data/research/genomics/projects/ppmi_analysis/proteomics_analysis/proteomic_VAE"
#python_path = "home/ubuntu/vae_proteins"
df = pd.read_csv(data_path + '/soma_matrix.csv', index_col=0)
df = df.reset_index()
df = df.rename(columns = {"index": "PATNO"})
df["PATNO"] = df["PATNO"].str.replace("S_", "").astype(int)
df = df.sort_values(by="PATNO")
protein_df = df[df["DxGx"]!="HC_NEG"]

disease_type = pd.DataFrame(protein_df[["PATNO", "DxGx"]])
disease_type.to_csv("files/disease_type.csv")


protein_df = protein_df.drop(columns = "DxGx")
protein_df = protein_df.set_index('PATNO')
protein_df = protein_df.add_prefix('Protein_')
protein_df.to_csv("files/soma_matrix_for_VAE.csv")


# View columns
#print(df.shape)
columns = protein_df.columns.to_list()
np.save("files/nb_patients.npy", protein_df.shape[0])
np.save("files/nb_proteins.npy", protein_df.shape[1])


std_df = protein_std(protein_df, columns)
std_df = std_df.sort_values(by = 'standard_deviation', ascending = False)
vals = np.array(std_df['standard_deviation'].values)
print(std_df)

plt.figure()
plt.plot(np.arange(len(std_df))+1, vals)
plt.title('Protein variation')
plt.xlabel('rank')
plt.ylabel('standard deviation')
plt.savefig('figs/01_standard_deviation.png')

import os
if os.path.exists('figs')==False:
    os.mkdir('figs')
if os.path.exists('files')==False:
    os.mkdir('files')
if os.path.exists('files/res')==False:
    os.mkdir('files/res')

"""
columns0 = protein_df.columns.to_list()
to_write = "{\n"
for ii, col in enumerate(columns):
    to_write = to_write + '"' + col + '"' + ": " + '"' + columns0[ii]+ '"' + ",\n"
to_write = to_write + "}"
f = open("files/colnames_for_plots.json", "w")
f.write(to_write)
f.close()"""