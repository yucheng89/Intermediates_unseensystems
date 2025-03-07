import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import pandas as pd
import time
from sklearn import metrics
import matplotlib as mpl
from sklearn.metrics import silhouette_score
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from umap import UMAP
from collections import Counter
from numpy import *
import torch

'''solvent_umap'''


def u_map(data,label,savepath):
    Umap = UMAP(n_neighbors=40, min_dist=1, n_components=2).fit_transform(data)
    for i in range(len(data)):
        if label[i]==0:
            s1 = plt.scatter(Umap[:, 0][i], Umap[:, 1][i],c='#afd4e3',s=5,alpha=0.9)
        if label[i]==1:
            s2 = plt.scatter(Umap[:, 0][i],Umap[:, 1][i],c='#c8c4e0',s=5,alpha=0.9)
        if label[i]==2:
            s3 = plt.scatter(Umap[:, 0][i], Umap[:, 1][i],c='#f5bfcb',s=5,alpha=0.9)
        if label[i]==3:
            s4 = plt.scatter(Umap[:, 0][i], Umap[:, 1][i],c='#d4d4d4',s=5,alpha=0.9)

    plt.legend((s1,s2,s3,s4),('No Solvent','Acetonitrile','DMF','Ethanol'),prop={'size': 9, 'family': 'Arial'}, markerscale=2, loc = 'lower right')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(savepath,dpi=800)

def load_onemetal_data(metal):
    data = pd.read_csv(f'{metal}.csv', header=None)
    return data

def load_onemetal_data_sol(metal,solvent):
    data = pd.read_csv(f'{metal}/{solvent}/data.csv', header=None)
    return data

def merge_data(metal_list):
    final_data = load_onemetal_data(metal_list[0])
    for i in metal_list[1:]:
        onedata = load_onemetal_data(i)
        final_data = pd.concat([final_data,onedata],axis=0)
    return final_data

def merge_data_sol(metal_list,solvent):
    final_data = load_onemetal_data_sol(metal_list[0],solvent)
    for i in metal_list[1:]:
        onedata = load_onemetal_data_sol(i,solvent)
        final_data = pd.concat([final_data,onedata],axis=0)
    return final_data

def handel_input(original_data):
    x_pre = original_data.iloc[:, :5000].values
    y = original_data.iloc[:, 5000:5007].values      # 0: CHO 1: CHOH 2: CO 3: COH 4: COOH 5: OCH3 6: OCHO
    y = torch.tensor(y, dtype=torch.float32)
    x_pre = torch.tensor(x_pre, dtype=torch.float32)
    original_net(x_pre)
    x_output = layer_output
    hook.remove()
    return x_output,y

from model import Batch_Net

original_net = torch.load('m_1.pkl')
original_net.eval()

# hook
features_layer = original_net.layer5
layer_output = None
def hook_fn(module, input, output):
    global layer_output
    layer_output = output
hook = features_layer.register_forward_hook(hook_fn)

all_metal_list = ['23V', '77Ir', '45Rh', '22Ti', '42Mo', '73Ta', '44Ru', '78Pt', '74W', '25Mn', '24Cr', '27Co', '75Re','76Os', '26Fe']

metal_list = all_metal_list[12:]  # unseen systems

No_sol = merge_data(metal_list)
acetonitrile_sol = merge_data_sol(metal_list, 'acetonitrile')
dmf_sol = merge_data_sol(metal_list, 'dmf')
ethanol_sol = merge_data_sol(metal_list, 'ethanol')

all_original_data = np.concatenate((No_sol,acetonitrile_sol,dmf_sol,ethanol_sol),axis=0)
all_original_data = pd.DataFrame(all_original_data)

all_feature = handel_input(all_original_data)[0]
all_feature = all_feature.detach().numpy()

labels = np.array([[0]*128*3,[1]*128*3,[2]*128*3,[3]*128*3])
labels = labels.flatten().reshape(-1,1)

original_spectrum = all_original_data.iloc[:, :5000].values

u_map(all_feature,labels,'umap3.png')
#u_map(original_spectrum,labels,'umap3.png')