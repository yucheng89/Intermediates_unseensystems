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

'''0/1'''

def u_map(data,label,savepath):
    Umap = UMAP(n_neighbors=40, min_dist=1, n_components=2).fit_transform(data)
    for i in range(len(data)):
        if label[i]==0:
            s1 = plt.scatter(Umap[:, 0][i], Umap[:, 1][i],c='#3083D8',s=10,alpha=0.7)
        if label[i]==1:
            s2 = plt.scatter(Umap[:, 0][i],Umap[:, 1][i],c='#FFBFCD',s=10,alpha=0.7)

    plt.legend((s1,s2),('Absence','Presence'),prop={'size': 10, 'family': 'Arial'}, markerscale=3, loc = 'lower left',framealpha=0.999)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(savepath,dpi=800)

def load_onemetal_data(metal):
    data = pd.read_csv(f'{metal}.csv', header=None)
    return data

def merge_data(metal_list):
    final_data = load_onemetal_data(metal_list[0])
    for i in metal_list[1:]:
        onedata = load_onemetal_data(i)
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

metal_list = ['23V', '77Ir', '45Rh', '22Ti', '42Mo', '73Ta', '44Ru', '78Pt', '74W', '25Mn', '24Cr', '27Co', '75Re', '76Os', '26Fe']
all_feature = handel_input(merge_data(metal_list))[0]
all_feature = all_feature.detach().numpy()

original_spectrum = merge_data(metal_list).iloc[:, :5000].values

all_y = merge_data(metal_list).iloc[:, 5000:5007].values  # 0: CHO 1: CHOH 2: CO 3: COH 4: COOH 5: OCH3 6: OCHO
specie = 0
y = all_y[:, specie]
labels = y

u_map(all_feature,labels,'umap2.png')
#u_map(original_spectrum,labels,'umap2.png')