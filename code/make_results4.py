
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import glob
import pandas as pd
import os
from itertools import compress
from results_utils import display_dataset_name, display_decoder_name, coef_variation, metrics, dict_mean, dict_ste, dict_mean_ste
from distinctipy import distinctipy
import pandas as pd


res_dir = Path("../work/results/") 
log_dir = "../work/out/"
latex_dir  = Path("../work/latex/")
pdfs_dir  = Path("../work/pdfs/")

latex_dir.mkdir(parents=True, exist_ok=True)
pdfs_dir.mkdir(parents=True, exist_ok=True)

configs_others = ["64", "64b", "64c"]
runs = [1, 2, 3]

tag = "real"

if tag == "real":
    datasets = ['yelp_airport', 'taxi', 'yelp_mississauga', 'twitter', 'wikipedia', 'pubg', 'yelp_toronto', 'reddit_askscience_comments', 'reddit_politics_submissions', 'lastfm', 'mooc', 'reddit']
    #datasets = ['yelp_airport', 'taxi', 'yelp_mississauga', 'twitter']
    #configs_rqs = ["1", "2", "3", "5", "8", "10", "15"]
    configs_rqs = ["1", "2", "3", "8", "10", "15"]


elif tag == "synth":
    datasets = ['synth/poisson', 'synth/renewal', 'synth/self_correcting', 'synth/hawkes2', 'synth/hawkes1']
    configs_rqs = ["3", "5", "8", "10", "15"]

elif tag == "both":
        datasets = ['synth/poisson', 'synth/renewal', 'synth/self_correcting', 'synth/hawkes2', 'synth/hawkes1', 'yelp_airport', 'taxi', 'yelp_mississauga', 'twitter', 'wikipedia', 'pubg', 'yelp_toronto', 'reddit_askscience_comments', 'reddit_politics_submissions', 'lastfm', 'mooc', 'reddit']


loss_test_mean_numbin = dict()
loss_val_mean_numbin = dict()
mace_test_mean_numbin = dict()

decoder = "RQS_EXP-crps_qapprox"
configs = configs_rqs


for dataset in datasets: ######
    print("--- ", dataset, " ---")

    dict_config_loss_val_mean = dict()
    dict_config_loss_test_mean = dict()
    dict_config_mace_test_mean = dict()
    for config in configs:

        list_loss_val_mean = []
        list_loss_test_mean = []
        list_mace_test_mean = []
        # validation
        for run in runs:
            suffix = str(dataset).replace("/", "-") + "-" + str(decoder)  + '-' + config + '-' + str(run) + '.pt'
            results_file = res_dir / suffix
            if results_file.exists():
                loaded = torch.load(results_file, map_location=torch.device('cpu'))

                list_loss_val_mean.append(loaded['loss_val'].item())
                list_loss_test_mean.append(loaded['loss_test'].item() )

                if  'predictions_test' in loaded.keys():
                    metrics_sampling = metrics(loaded["predictions_test"], mace_only = True)
                    list_mace_test_mean.append(metrics_sampling["mace"])
                else:
                    print("Predictions missing in ", results_file, " !!!!!")
                    list_mace_test_mean.append(np.inf)


                #all_metrics_sampling_best.append(metrics_sampling_best)
            
            else:
                print(results_file, " missing!")

        print("****")
        print(list_loss_val_mean)
        print("---")
        print(list_loss_test_mean)
        print("---")
        print(list_mace_test_mean)

        #loss_val = np.mean(list_loss_val)
        #loss_test = np.mean(list_loss_test)
        #mace_test = np.mean(list_mace_test)

        #dict_config_val[config] = "{:0.3f}".format(np.mean(list_loss_val)) + " (" + "{:0.3f}".format(np.std(list_loss_val)/np.sqrt(len(list_loss_val)) ) + ")"
        dict_config_loss_val_mean[config] = "{:0.3f}".format(np.mean(list_loss_val_mean)) 
        
        #dict_config_test[config] = "{:0.3f}".format(np.mean(list_loss_test)) + " (" + "{:0.3f}".format(np.std(list_loss_test)/np.sqrt(len(list_loss_test)) ) + ")"
        dict_config_loss_test_mean[config] = "{:0.3f}".format(np.mean(list_loss_test_mean)) 

        #dict_config_mace_test[config] = "{:0.3f}".format(np.mean(list_mace_test)) + " (" + "{:0.3f}".format(np.std(list_mace_test)/np.sqrt(len(list_mace_test)) ) + ")"
        dict_config_mace_test_mean[config] = "{:0.3f}".format(np.mean(list_mace_test_mean)) 


    loss_val_mean_numbin[dataset] = dict_config_loss_val_mean
    loss_test_mean_numbin[dataset] = dict_config_loss_test_mean
    mace_test_mean_numbin[dataset] = dict_config_mace_test_mean


loss_val_mean_numbin = pd.DataFrame.from_dict(loss_val_mean_numbin, orient='index')
loss_test_mean_numbin = pd.DataFrame.from_dict(loss_test_mean_numbin, orient='index')
mace_test_mean_numbin = pd.DataFrame.from_dict(mace_test_mean_numbin, orient='index')

table_loss_mace = loss_test_mean_numbin + " (" + mace_test_mean_numbin + ")"

loss_val_mean_numbin.rename(display_dataset_name, axis = 0, inplace = True)
loss_test_mean_numbin.rename(display_dataset_name, axis = 0, inplace = True)
mace_test_mean_numbin.rename(display_dataset_name, axis = 1, inplace = True)

###
table_file = "table_numbin_loss_" + tag + ".tex"
results_file = latex_dir / table_file
with open(results_file, 'w') as tf:
     res = loss_test_mean_numbin.to_latex(float_format="{:0.3f}".format, index_names = False)
     print(res)
     tf.write(res)

table_file = "table_numbin_mace_" + tag + ".tex"
results_file = latex_dir / table_file
with open(results_file, 'w') as tf:
     res = mace_test_mean_numbin.to_latex(float_format="{:0.3f}".format, index_names = False)
     print(res)
     tf.write(res)

if False:
    table_file = "table_numbin_loss_mace_" + tag + ".tex"
    results_file = latex_dir / table_file
    with open(results_file, 'w') as tf:
        res = table_loss_mace.to_latex(float_format="{:0.3f}".format, index_names = False)
        print(res)
        tf.write(res)

breakpoint()

