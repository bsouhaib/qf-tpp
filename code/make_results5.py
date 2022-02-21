import dpp
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from pathlib import Path

import pprint
import time
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from make_parser import myparser
from config import update_args, print_args

pdfs_dir  = Path("../work/pdfs/")

#dataset_dir = Path(__file__).parents[2] / 'data'
dataset_dir = Path("/home/ubuntu/tpp-rqs-crps/data")


#'synth/poisson' 'synth/renewal' 'synth/self_correcting' 'synth/hawkes2' 'synth/hawkes1'
#for dataset in 'yelp_airport' 'taxi' 'yelp_mississauga' #'twitter' 'wikipedia' 'pubg'  'yelp_toronto'  'reddit_askscience_comments' 'reddit_politics_submissions' 'lastfm' 'mooc' 'reddit' 

name = 'reddit.npz' 
name = 'yelp_toronto.npz' 
name = 'synth/hawkes1.npz' 
name = 'lastfm.npz'
name = 'wikipedia.npz' 



do_log = False

if False:
    dataset_dir_new = dataset_dir / 'data-triangular'
    loader = torch.load(dataset_dir_new / name, map_location=torch.device('cpu'))
    sequences = loader["sequences"]
else:
    loader = dict(np.load(dataset_dir / name, allow_pickle=True))
    sequences = loader['arrival_times']

if False:
    n_sequences = 20
    id_sequences = np.arange(n_sequences)
else:
    my_ids = [603, 609, 702, 707, 712, 713, 720, 721, 722, 723]
    # my_ids = np.arange(720, 730)
    id_sequences = np.argsort([len(seq) for seq in sequences])[my_ids] # 
    n_sequences = len(id_sequences)



fig, axes = plt.subplots(n_sequences, 1, figsize=(8.27, 4))

#ax = fig.add_subplot(10, 1)

all_times = []
for idx in id_sequences:
    sequence = sequences[idx] #[:100]
    if do_log:
        sequence = [np.log(1 + x) for x in sequence]
    all_times.append(sequence)
    
#all_times = [item for sublist in all_times for item in sublist]
#t_min = np.min(all_times)
#t_max = np.max(all_times)

all_times = [item for sublist in all_times for item in sublist]

for idx in id_sequences:
    seq = sequences[idx]
    sequences[idx] = [(x - np.min(all_times))/(np.max(all_times) - np.min(all_times))  for x in  seq] 

t_min = 0
t_max = 1


for i, idx in enumerate(id_sequences):
    sequence = sequences[idx]  #[:100]
    if do_log:
        sequence = [np.log(1 + x) for x in sequence]

    #axes[idx].plot(sequence, np.zeros(len(sequence)) + idx + 5, '|', ms=20)  # rug plot
    axes[i].plot(sequence, np.zeros(len(sequence)), '|', ms=20)  # rug plot

    axes[i].set_xlim(left = t_min, right = t_max)

    if i != (n_sequences-1):
        axes[i].set_yticks([])
        axes[i].set_xticks([])
    if i == (n_sequences-1):
        axes[i].set_yticks([])
        #axes[i].set_xlabel("Time")
    
    
    #["{:0.1f}".format(x) for x in sequences[idx]]
    #plt.xticks(sequence, sequences[idx], rotation='vertical')

    #plt.xticks([t_min, t_max], [t_min, t_max], rotation='vertical')

#x_eval = np.linspace(-10, 10, num=200)
#ax.plot(x_eval, kde1(x_eval), 'k-', label="Scott's Rule")
#ax.plot(x_eval, kde1(x_eval), 'r-', label="Silverman's Rule")

#import seaborn as sns; sns.set_theme()
#tips = sns.load_dataset("tips")
#sns.rugplot(x = np.arange(len(sequence)), y = sequence)

    plt.show()

#plt.axis('off')
#fig.axes.get_xaxis().set_visible(False)

fig.savefig(pdfs_dir / ("rug.pdf") , bbox_inches='tight')



breakpoint()