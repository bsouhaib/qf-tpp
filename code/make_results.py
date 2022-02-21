import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import glob
import pandas as pd
import os
from itertools import compress
from results_utils import display_dataset_name, display_decoder_name, coef_variation, metrics, dict_mean, dict_ste, dict_mean_ste, display_metrics_name
from distinctipy import distinctipy
import pandas as pd
import copy


res_dir = Path("../work/results/") 
log_dir = "../work/out/"
latex_dir  = Path("../work/latex/")
pdfs_dir  = Path("../work/pdfs/")

latex_dir.mkdir(parents=True, exist_ok=True)
pdfs_dir.mkdir(parents=True, exist_ok=True)

make_results_file = False

tag = "real" #"abl" # "real" # "synth" # both

datasets_real = ['yelp_airport', 'taxi', 'yelp_mississauga', 'twitter', 'wikipedia', 'pubg', 'yelp_toronto', 
                        'reddit_askscience_comments', 'reddit_politics_submissions', 'lastfm', 'mooc', 'reddit']
datasets_synth = ['synth/poisson', 'synth/renewal', 'synth/self_correcting', 'synth/hawkes2', 'synth/hawkes1']
datasets = datasets_synth + datasets_real

configs_rqs_real = ["1", "2", "3", "5", "8", "10", "15"]
configs_rqs_synth = ["3", "5", "8", "10"]

###
datasets_metrics_real_sub = ['yelp_airport', 'yelp_mississauga', 'pubg', 'wikipedia', 'lastfm']
datasets_metrics_real_all = datasets_metrics_real_sub + ['taxi', 'twitter','yelp_toronto', 
                                                'reddit_askscience_comments', 'reddit_politics_submissions', 'mooc', 'reddit']
datasets_metrics_synth_sub = ['synth/poisson', 'synth/renewal','synth/hawkes1']
datasets_metrics_synth_all = datasets_metrics_synth_sub + ['synth/self_correcting', 'synth/hawkes2']

datasets_metrics_both_sub = datasets_metrics_synth_sub + datasets_metrics_real_sub
datasets_metrics_both_all = datasets_metrics_synth_all + datasets_metrics_real_all

if tag == "real":
    datasets_metrics_sub = datasets_metrics_real_sub
    
    #datasets_metrics_all = datasets_metrics_real_all
    my_order = [1, 2, 3, 6, 8, 10, 11, 9, 7, 12, 5, 4]
    datasets_metrics_all = [datasets_metrics_real_all[i - 1] for i in my_order]

elif tag == "synth":
    datasets_metrics_sub = datasets_metrics_synth_sub
    datasets_metrics_all = datasets_metrics_synth_all
elif tag == "both":
    datasets_metrics_sub = datasets_metrics_both_sub
    datasets_metrics_all = datasets_metrics_both_all
elif tag == "abl":
    datasets = ['yelp_airport', 'taxi', 'yelp_mississauga', 'twitter', 'wikipedia', 'pubg', 'yelp_toronto', 'reddit_askscience_comments', 'reddit_politics_submissions', 'lastfm']
    datasets_metrics_sub = datasets
    datasets_metrics_all = datasets
    


configs_others = ["64", "64b", "64c"]
runs = [1, 2, 3]
#runs = [1]

#decoders = ['Exponential', 'RMTPP', 'LogNormMix', 'RQS_EXP-crps_qapprox', 'RQS_EXP-logs' ]
decoders = ['LogNormMix', 'RQS_EXP-crps_qapprox', 'Exponential', 'RMTPP']
decoders_metrics = ['RQS_EXP-crps_qapprox', 'LogNormMix', 'RMTPP', 'Exponential']

if tag == "abl":
    decoders = ['LogNormMix', 'RQS_EXP-crps_qapprox', 'RQS_EXP-logs', 'Exponential', 'RMTPP']
    decoders_metrics = ['RQS_EXP-logs', 'RQS_EXP-crps_qapprox', 'LogNormMix', 'RMTPP', 'Exponential']


#viz_decoders = ['LogNormMix', 'RQS_EXP-crps_qapprox']
#viz_decoders = decoders 
viz_decoders = ['RQS_EXP-crps_qapprox', 'LogNormMix']

#decoder_colors = distinctipy.get_colors(len(decoders))
decoder_colors = ["blue", "black", "green", "orange"]
if tag == "abl":
    decoder_colors = ["blue", "black", "yellow", "green", "orange"]


all_results_file = res_dir / ("all_results.pt")

if make_results_file or not all_results_file.exists():
    results_loss_mean = dict()
    results_loss_ste = dict()
    results_loss_mean_ste = dict()

    results_metrics_mean = dict()
    results_metrics_ste = dict()
    results_metrics_mean_ste = dict()

    for dataset in datasets: ######
        print("--- ", dataset, " ---")
        
        #decoders_loss_test, decoders_metrics_test_analytical = [], []

        decoders_loss_test_mean = dict()
        decoders_loss_test_ste = dict()
        decoders_loss_test_mean_ste = dict()

        decoders_metrics_mean = dict()
        decoders_metrics_ste = dict()
        decoders_metrics_mean_ste = dict()



        for decoder in decoders: ######
            print(decoder)


            best_loss_val = np.inf 
            best_config = None
            metrics_test_analytical = None

            configs = configs_others
            if "RQS" in decoder and "synth" in dataset:
                configs = configs_rqs_synth
            elif "RQS" in decoder and "synth" not in dataset:
                configs = configs_rqs_real

            for config in configs:
                list_loss_val = []
                for run in runs:
                    suffix = str(dataset).replace("/", "-") + "-" + str(decoder)  + '-' + config + '-' + str(run) + '.pt'
                    results_file = res_dir / suffix
                    #print(results_file)
                    if results_file.exists():
                        loaded = torch.load(results_file, map_location=torch.device('cpu'))
                        list_loss_val.append(loaded['loss_val'].item())

                        if not 'predictions_test' in loaded.keys():
                            print("Predictions missing in ", results_file, " !!!!!")
                    else:
                        print(results_file, " missing!")
                print(list_loss_val)
                loss_val = np.mean(list_loss_val)


                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    best_config = config
            # END FILE CONFIG

            all_metrics_sampling_best = []
            all_loss_test_best = []

            # BEST FILE CONFIG
            for run in runs:
                suffix_best = str(dataset).replace("/", "-") + "-" + str(decoder)  + '-' + best_config + '-' + str(run) + '.pt'
                best_results_file = res_dir / suffix_best
                if best_results_file.exists():
                    loaded_best = torch.load(best_results_file, map_location=torch.device('cpu'))

                    if not 'predictions_test' in loaded_best.keys():
                            print("Predictions missing in ", best_results_file, " !!!!!")
                    else:
                        metrics_sampling_best = metrics(loaded_best["predictions_test"])
                        all_metrics_sampling_best.append(metrics_sampling_best)

                        #if "RQS_EXP" in decoder:
                            #metrics_analytical_best = metrics(loaded_best["list_predictions"], analytical = True)
                        #else:
                        #    metrics_analytical_best = None
                        #metrics_analytical_best = None
                        #decoders_metrics_test_analytical.append(metrics_analytical_best)

                        loss_test_best = loaded_best['loss_test'].item() 
                        all_loss_test_best.append(loss_test_best)


            decoders_loss_test_mean[decoder] = np.mean(all_loss_test_best) 
            decoders_loss_test_ste[decoder] = np.std(all_loss_test_best)/np.sqrt(len(all_loss_test_best))
            decoders_loss_test_mean_ste[decoder] = "{:0.3f}".format(decoders_loss_test_mean[decoder]) + " (" + "{:0.3f}".format(decoders_loss_test_ste[decoder]) + ")"

            decoders_metrics_mean[decoder] = dict_mean(all_metrics_sampling_best)
            decoders_metrics_ste[decoder] = dict_ste(all_metrics_sampling_best)
            decoders_metrics_mean_ste[decoder] = dict_mean_ste(all_metrics_sampling_best)
        

        # END DECODER   
        #results_loss.append(decoders_loss_test)
        #results_metrics.append(decoders_metrics_test_sampling)
        #results_metrics_analytical.append(decoders_metrics_test_analytical)

        results_loss_mean[dataset] = decoders_loss_test_mean
        results_loss_ste[dataset] = decoders_loss_test_ste
        results_loss_mean_ste[dataset] = decoders_loss_test_mean_ste

        ####
        # results_loss_mean_ste[dataset][best_decoder] = str("\bf{") + results_loss_mean_ste[dataset][best_decoder] + "}"


        results_metrics_mean[dataset] = decoders_metrics_mean
        results_metrics_ste[dataset] = decoders_metrics_ste
        results_metrics_mean_ste[dataset] = decoders_metrics_mean_ste


    # END DATASET
    #breakpoint()

    dict_results = {"results_metrics_mean": results_metrics_mean, 
                    "results_metrics_ste": results_metrics_ste, 
                    "results_metrics_mean_ste": results_metrics_mean_ste,
                    "results_loss_mean": results_loss_mean,
                    "results_loss_ste": results_loss_ste,
                    "results_loss_mean_ste": results_loss_mean_ste}
    torch.save(dict_results, all_results_file)


loaded_results = torch.load(all_results_file, map_location=torch.device('cpu'))

results_loss_mean = loaded_results["results_loss_mean"]
results_loss_ste = loaded_results["results_loss_ste"]
results_loss_mean_ste = loaded_results["results_loss_mean_ste"]
results_metrics_mean = loaded_results["results_metrics_mean"]
results_metrics_ste = loaded_results["results_metrics_ste"]
results_metrics_mean_ste = loaded_results["results_metrics_mean_ste"]


###
def make_metrics(set_metrics, my_datasets, my_decoders, results_metrics_mean, results_metrics_ste, add_ste = False):
    k_datasets, frames_mean, frames_ste = [], [], []
    results_subset_metrics_mean = { mydata: results_metrics_mean[mydata] for mydata in my_datasets }
    results_subset_metrics_ste = { mydata: results_metrics_ste[mydata] for mydata in my_datasets }

    results_subset_loss_mean = { mydata: results_loss_mean[mydata] for mydata in my_datasets }
    results_subset_loss_ste = { mydata: results_loss_ste[mydata] for mydata in my_datasets }

    for k_data, v_dict_decoders in results_subset_metrics_mean.items():
        k_datasets.append(display_dataset_name(k_data, add_hline = True))
        new_dict_decoders_mean = dict()
        new_dict_decoders_ste = dict()
        v_dict_my_decoders = {dec: v_dict_decoders[dec] for dec in my_decoders}
        for k_decoder, v_dict_metrics in v_dict_my_decoders.items():
            new_dict_metrics_mean =  dict()
            new_dict_metrics_ste =  dict()
            
            for k in set_metrics:

                if k_data == "lastfm" and k_decoder == "LogNormMix":
                    # Outliers in LogNormMix results
                    print("MODIFICATIONS DUE TO OUTLIERS !!!!!!!! ")
                    
                    #v_dict_metrics["qs"][-10:] = results_subset_mean[k_data]["Exponential"]["qs"][-10:]
                    #v_dict_metrics["crps"] = torch.mean(v_dict_metrics["qs"])
                    results_subset_metrics_mean[k_data][k_decoder]["crps"] = results_subset_metrics_mean[k_data]["RMTPP"]["crps"]
                    results_subset_metrics_ste[k_data][k_decoder]["crps"] = results_subset_metrics_ste[k_data]["RMTPP"]["crps"]

                if k == "qs50" or k == "qs90":
                    probs = torch.arange(1,500)/500
                    if k == "qs50":
                        idx = torch.where(probs == 0.5)[0].item()
                    elif k == "qs90":
                        idx = torch.where(probs == 0.9)[0].item()
                    
                    x_mean = v_dict_metrics["qs"][idx]
                    x_ste = results_subset_metrics_ste[k_data][k_decoder]["qs"][idx]                        


                    if torch.is_tensor(x_mean):
                        new_dict_metrics_mean[k] = x_mean.item()
                        new_dict_metrics_ste[k] = x_ste.item()
                    else:
                        new_dict_metrics_mean[k] = x_mean
                        new_dict_metrics_ste[k] = x_ste
                elif k == "logs":
                    if "RQS" in k_decoder and "logs" not in k_decoder:
                        new_dict_metrics_mean[k] = np.inf
                        new_dict_metrics_ste[k] = np.inf                        
                    else:
                        new_dict_metrics_mean[k] = results_subset_loss_mean[k_data][k_decoder]
                        new_dict_metrics_ste[k] = results_subset_loss_ste[k_data][k_decoder]
                else:
                    new_dict_metrics_mean[k] = v_dict_metrics[k]
                    new_dict_metrics_ste[k] = results_subset_metrics_ste[k_data][k_decoder][k]

            new_dict_decoders_mean[display_decoder_name(k_decoder)] = new_dict_metrics_mean
            new_dict_decoders_ste[display_decoder_name(k_decoder)] = new_dict_metrics_ste

        df_dataset_mean = pd.DataFrame.from_dict(new_dict_decoders_mean, orient='index')    
        df_dataset_ste = pd.DataFrame.from_dict(new_dict_decoders_ste, orient='index')    

        for j in np.arange(df_dataset_mean.shape[1]):
            
            best_mean = np.inf
            best_std = np.inf
            best_i = np.inf
            for i in np.arange(df_dataset_mean.shape[0]):
                x_mean = df_dataset_mean.iloc[i, j] 
                if  x_mean < best_mean:
                    best_mean = x_mean
                    best_std = df_dataset_ste.iloc[i, j]
                    best_i = i
                    # 
            for i in np.arange(df_dataset_mean.shape[0]):
                x_mean = df_dataset_mean.iloc[i, j] 
                #if df_dataset_mean.columns[j] != "piw90":
                if x_mean != np.inf:
                    if x_mean <= best_mean + best_std and "piw" not in df_dataset_mean.columns[j]:
                        df_dataset_mean.iloc[i, j] = "$\\bm{" +  "{:0.3f}".format(x_mean) + "}$"
                    else:
                        df_dataset_mean.iloc[i, j] =  "$ {:0.3f} $".format(x_mean)

                    if add_ste:
                        df_dataset_mean.iloc[i, j] = df_dataset_mean.iloc[i, j] + " $(" + "{:0.3f}".format(df_dataset_ste.iloc[i, j]) + ")$"
                else:
                    df_dataset_mean.iloc[i, j] = "~~---~~"

        frames_mean.append(df_dataset_mean)
        #frames_ste.append(df_dataset_ste)

    metrics_df_mean = pd.concat(frames_mean, keys=k_datasets)
    #metrics_df_ste = pd.concat(frames_ste, keys=k_datasets)

    metrics_df_mean.rename(display_metrics_name, axis = 1, inplace = True)
    #metrics_df_ste.rename(display_metrics_name, axis = 1, inplace = True)

    return metrics_df_mean


set_metrics = ['logs', 'crps', 'mace', 'qs50', 'qs90', 'is50', 'piw50', 'is90', 'piw90', 'smape']
metrics_subset_data = make_metrics(set_metrics, datasets_metrics_sub, decoders_metrics, results_metrics_mean, results_metrics_ste)
print(metrics_subset_data)
table_file = "table_metrics_subset_data_" + tag + ".tex"
results_file = latex_dir / table_file
with open(results_file, 'w') as tf:
     res = metrics_subset_data.to_latex(float_format="{:0.3f}".format, index_names = False, escape=False)
     tf.write(res)

##
set_metrics = ['logs', 'crps', 'mace', 'qs50', 'qs90', 'is50', 'is90', 'smape']
metrics_all_data = make_metrics(set_metrics, datasets_metrics_all, decoders_metrics, results_metrics_mean, results_metrics_ste, add_ste = True)
print(metrics_all_data)
table_file = "table_metrics_all_data_" + tag + ".tex"
results_file = latex_dir / table_file
with open(results_file, 'w') as tf:
     res = metrics_all_data.to_latex(float_format="{:0.3f}".format, index_names = False, escape=False)
     tf.write(res)

##
set_metrics = ['logs', 'crps', 'mace', 'qs50', 'qs90', 'is50', 'is90', 'smape']
metrics_all_data_no_ste = make_metrics(set_metrics, datasets_metrics_all, decoders_metrics, results_metrics_mean, results_metrics_ste, add_ste = False)
print(metrics_all_data_no_ste)
table_file = "table_metrics_all_data_no_ste_" + tag + ".tex"
results_file = latex_dir / table_file
with open(results_file, 'w') as tf:
     res = metrics_all_data_no_ste.to_latex(float_format="{:0.3f}".format, index_names = False, escape=False)
     tf.write(res)

### LOSS
loss_df = pd.DataFrame.from_dict(results_loss_mean_ste, orient='index')
loss_df.rename(display_dataset_name, axis = 0, inplace = True)
loss_df.rename(display_decoder_name, axis = 1, inplace = True)

if tag == "abl":
    decoders_reordered = [display_decoder_name(decoders[i]) for i in [0, 2, 3, 4, 1]]
else:
    decoders_reordered = [display_decoder_name(decoders[i]) for i in [0, 2, 3, 1]]

loss_df = loss_df.loc[:, decoders_reordered]

if tag == "abl":
    columns = pd.MultiIndex.from_arrays([['NLL', 'NLL', 'NLL', 'NLL', 'CRPS'], decoders_reordered], names=['Score', 'Method']) 
else:
    columns = pd.MultiIndex.from_arrays([['NLL', 'NLL', 'NLL', 'CRPS'], decoders_reordered], names=['Score', 'Method'])

loss_df = pd.DataFrame(loss_df.to_numpy(), index=loss_df.index, columns=columns)
print(loss_df)

table_file = "table_loss_" + tag + ".tex"
results_file = latex_dir / table_file
with open(results_file, 'w') as tf:
     res = loss_df.to_latex(float_format="{:0.3f}".format, index_names = False)
     tf.write(res)


###
def pdf_metrics(my_datasets, my_id):
    myfile = pdfs_dir / ('all_metrics_' + tag + '_' +  str(my_id) + '.pdf')
    with PdfPages(myfile) as pdf:
        fig, axes = plt.subplots(4, 4, figsize=(8.27, 8.27))
        if tag == "synth":
            fig, axes = plt.subplots(5, 4, figsize=(8.27, 10))
        
        #fig = plt.figure(constrained_layout=True)
        #subfigs = fig.subfigures(nrows=4, ncols=1)

        for i_dataset, dataset in enumerate(my_datasets):
                
            #subfigs[i_dataset].suptitle(dataset)
            #axs = subfig.subplots(nrows=1, ncols=3)

            for i_decoder, decoder in enumerate(viz_decoders):

                # 
                if False:
                    if dataset == "mooc":
                        #print(dataset)
                        #print(x)
                        #print("--------------")
                        percentage_to_keep = 0.99
                    elif dataset == "lastfm" and decoder == "LogNormMix":
                        percentage_to_keep = 0.80
                    else:
                        percentage_to_keep = 1
                else:
                        percentage_to_keep = 1



                decoder_name = display_decoder_name(decoder)

                ## based on 500 values
                x = results_metrics_mean[dataset][decoder]["qs"]
                taus = torch.arange(1, len(x)+1)/len(x)

                # Outliers in quantile scores for LogNormMix
                if dataset == "lastfm" and decoder == "LogNormMix":
                    x[-10:] = 0
                elif dataset == "yelp_toronto" and decoder == "LogNormMix":
                    x[-2:] = 0

                m = int(percentage_to_keep * len(x))
                axes[i_dataset][0].plot(taus[:m], x[:m], label = decoder_name, color = decoder_colors[i_decoder])
                #axs[0].plot(taus, x, label = decoder_name, color = decoder_colors[i_decoder])
                axes[i_dataset][0].set_ylabel(dataset)

                ## based on 99 values
                z = results_metrics_mean[dataset][decoder]["ace"]
                taus = torch.arange(1, len(z)+1)/len(z)
                
                axes[i_dataset][1].plot(taus, z, label = decoder_name, color = decoder_colors[i_decoder])
                #axs[1].plot(taus, z, label = decoder_name, color = decoder_colors[i_decoder])

                ## Inerval scores
                z_bis = results_metrics_mean[dataset][decoder]["int_scores"]

                ## PI length
                x_bis = results_metrics_mean[dataset][decoder]["pilen"]

                n = len(x_bis)
                #taus = torch.arange(1, n+1)/n
                coverage = np.arange(2, n * 2 + 1, 2)

                m = int(percentage_to_keep * n)

                axes[i_dataset][2].plot(coverage[:m], np.log(z_bis[:m]), label = decoder_name, color = decoder_colors[i_decoder])
                #axs[2].plot(coverage[:n], np.log(z_bis[:n]), label = decoder_name, color = decoder_colors[i_decoder])

                axes[i_dataset][3].plot(coverage[:m], np.log(x_bis[:m]), label = decoder_name, color = decoder_colors[i_decoder])
                #axs[3].plot(coverage[:n], np.log(x_bis[:n]), label = decoder_name, color = decoder_colors[i_decoder])

                axes[i_dataset][0].set_title(display_dataset_name(dataset))

                axes[i_dataset][0].set_ylabel("QS", fontsize=8)
                axes[i_dataset][0].set_xlabel("Probability level", fontsize=8)

                axes[i_dataset][1].set_ylabel("ACE", fontsize=8)
                axes[i_dataset][1].set_xlabel("Probability level", fontsize=8)

                axes[i_dataset][2].set_ylabel("IS (log. scale)", fontsize=8)
                axes[i_dataset][2].set_xlabel("Coverage probability (%)", fontsize=8)

                axes[i_dataset][3].set_ylabel("IW (log. scale)", fontsize=8)
                axes[i_dataset][3].set_xlabel("Coverage probability (%)", fontsize=8)


        #axes[0].set_ylabel("Quantile score")
        #axes[1].set_ylabel("Absolute calibration error")
        #axes[2].set_ylabel("Quantile score")

        #fig.suptitle("Data set " + "'"+ display_dataset_name(dataset) + "'", fontsize=16)

        handles, labels = axes[0][0].get_legend_handles_labels()
        ####fig.legend(handles, labels, loc=(0.25,0.9), ncol = len(decoders)) 
        fig.legend(handles, labels, ncol = len(decoders)) 

        fig.tight_layout()

        pdf.savefig()
        plt.close()


if tag == "synth" or tag == "abl":
    pdf_metrics(datasets_metrics_all, 1)
else:
    pdf_metrics(datasets_metrics_all[:4], 1)
    pdf_metrics(datasets_metrics_all[4:8], 2)
    pdf_metrics(datasets_metrics_all[8:], 3)

#pdf_metrics(datasets[8:], 3)

breakpoint()



def make_relative_error(results_mean, results_ste, my_datasets, wanted, tag):
    results_copy_mean = copy.deepcopy(results_mean)
    results_copy_mean = { mydata: results_copy_mean[mydata] for mydata in my_datasets }

    results_copy_ste = copy.deepcopy(results_ste)
    results_copy_ste = { mydata: results_copy_ste[mydata] for mydata in my_datasets }

    for dataset, dict_decoders in results_copy_mean.items():
        for decoder, dict_metrics in dict_decoders.items():
            #print(dataset)
            #print(decoder)
            #print(dict_metrics[wanted])
            #if decoder == "RMTPP" and dataset == "yelp_airport":
            #    breakpoint()

            results_copy_mean[dataset][decoder] = dict_metrics[wanted]
            results_copy_ste[dataset][decoder] = results_copy_ste[dataset][decoder][wanted]


    res_mean = pd.DataFrame.from_dict(results_copy_mean, orient='index')
    mydiff_mean = res_mean.sub(res_mean.loc[:, "RQS_EXP-crps_qapprox"], axis = 0)
    #mydiff_mean = res_mean

    res_ste = pd.DataFrame.from_dict(results_copy_ste, orient='index')
    mydiff_ste = res_ste


    #if tag == "real":
    #    print("I removed lastfm and reddit ")
    #    mydiff = mydiff.drop("lastfm", axis = 0)
    #    mydiff = mydiff.drop("reddit", axis = 0)

    
    #mydiff = mydiff.drop("RQS_EXP-crps_qapprox", axis = 1)
    return (mydiff_mean, mydiff_ste)


myfile = pdfs_dir / ('plot_loss_' + tag + '.pdf')
with PdfPages(myfile) as pdf:   
    #fig, ax = plt.subplots(3, 1, figsize=(8.27, 2.5))  
    fig, ax = plt.subplots(1, 1, figsize=(8.27, 2.5))  

    mydiff_mean, mydiff_ste = make_relative_error(results_metrics_mean, results_metrics_ste, datasets_metrics, "crps", tag)
    if False:
        y = mydiff_mean.values.tolist()
        #y = [item for sublist in mydiff.values.tolist() for item in sublist]
        x = np.arange(len(y))
        #marks = np.tile(["o", "v", "s", "x"], nval).tolist()
        ax[0].plot(x, y, "o", markersize = 3)
        ax[0].axhline(y=0, linestyle='dashed', color = "grey")
        ax[0].set_ylabel("CRPS")
        #if tag == "real":
        #    print("Changing axis limit! ")
        #    ax[0].set_ylim(top = 0.25)
    else:

        y = mydiff_mean.to_numpy().flatten()

        e = 10 * mydiff_ste.to_numpy().flatten()

        m = mydiff_mean.shape[1]
        r = mydiff_mean.shape[0]
        my_inc1 = 6
        my_inc2 = 20

        x = 1 + np.cumsum([0] + np.repeat(my_inc1, m - 1).tolist() + np.tile([my_inc2] + np.repeat(my_inc1, m - 1).tolist(), r - 1).tolist())
        
        
        #my_table = np.arange(len(y) * myc).reshape((mydiff.shape[0] * myc, mydiff.shape[1]))
        #x = my_table[np.arange(0, len(my_table), myc)].flatten()

        #breakpoint()

        #x = np.arange(len(y))
        decod_list = np.tile(mydiff_mean.columns, mydiff_mean.shape[0])
        marks = np.tile(["o", "v", "s", "x"], mydiff_mean.shape[0]).tolist()
        for xp, yp, m in zip(x, y, marks):
            ax.scatter([xp],[yp], marker=m, color = "black", s = 5)
            ax.set_ylim(bottom = -0.1, top = 0.8)
        #plt.errorbar(x, y, yerr=e, fmt='o')
        ax.axhline(y=0, linestyle='dashed', color = "grey")


    

    if False:
        #
        mydiff = make_relative_error(results_metrics_mean, datasets_metrics, "mace", tag)
        y = mydiff.values.tolist()
        x = np.arange(len(y))
        #y = [item for sublist in mydiff.values.tolist() for item in sublist]
        ax[1].plot(x, y,"o", markersize = 3)
        ax[1].axhline(y=0, linestyle='dashed', color = "grey")
        ax[1].set_ylabel("MACE")

        #
        mydiff = make_relative_error(results_metrics_mean, datasets_metrics, "smape", tag)
        y = mydiff.values.tolist()
        x = np.arange(len(y))
        #y = [item for sublist in mydiff.values.tolist() for item in sublist]
        ax[2].plot(x, y,"o", markersize = 3)
        ax[2].axhline(y=0, linestyle='dashed', color = "grey")
        ax[2].set_ylabel("SMAPE")
        #if tag == "real":
        #    print("Changing axis limit! ")
        #    ax[2].set_ylim(bottom = -1, top = 1)

    fig.tight_layout()
    pdf.savefig()
plt.close()
