import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import glob
import pandas as pd
import os

from results_utils import display_dataset_name, display_decoder_name, coef_variation, metrics, dict_mean, dict_ste, dict_mean_ste, display_metrics_name

res_dir = Path("../work/results/") 
latex_dir  = Path("../work/latex/")

#data_sets = ['synth/poisson', 'synth/renewal', 'synth/self_correcting', 'synth/hawkes2', 'synth/hawkes1']
dataset = 'synth/poisson'
decoders = ['RQS_EXP-crps', 'RQS_EXP-crps_qapprox_100', 'RQS_EXP-crps_qapprox_200', 'RQS_EXP-crps_qapprox_500']

run = 1
configs_numbers = [1, 2, 3, 5, 8, 10, 15]
configs = [str(cfg) + "_time" for cfg in configs_numbers]

decoders_forward, decoders_backwards, decoders_both  = [], [], []
for decoder in decoders: ######
    res_forward, res_backwards, res_both  = [], [], []
    for config in configs:
        suffix = str(dataset).replace("/", "-") + "-" + str(decoder) + '-' +  str(config) + '-' + str(run)
        execution_time_file = res_dir / (suffix + 'execution_time.pt')

        loaded = torch.load(execution_time_file, map_location=torch.device('cpu'))
        res = [ (b-a, c-b) for a, b, c in zip(loaded["start_time_forward"], loaded["start_time_backward"], loaded["end_time_backward"])]

        forward_times = [b-a for a, b in zip(loaded["start_time_forward"], loaded["start_time_backward"])]
        backward_times = [c-b for b, c in zip(loaded["start_time_backward"], loaded["end_time_backward"])]
        sum_times = [f+b for f, b in zip(forward_times, backward_times)] 

        print(np.mean(forward_times), np.std(forward_times))
        print(np.mean(backward_times), np.std(backward_times))
        print(np.mean(sum_times), np.std(sum_times))
        print("---")


        x_forward = "{:0.3f}".format(np.mean(forward_times)) + " (" + "{:0.3f}".format(np.std(forward_times)) + ")" 
        x_backwards = "{:0.3f}".format(np.mean(backward_times)) + " (" + "{:0.3f}".format(np.std(backward_times)) + ")"

        x_both = "{:0.3f}".format(np.mean(sum_times)) + " (" + "{:0.3f}".format(np.std(sum_times)) + ")"

        res_forward.append(x_forward)
        res_backwards.append(x_backwards)
        res_both.append(x_both)

    decoders_forward.append(res_forward)
    decoders_backwards.append(res_backwards)
    decoders_both.append(res_both)


#breakpoint()

#index_data =  [display_dataset_names(x) for x in data_sets]
decoders_newname = [display_decoder_name(x) for x in decoders]
#decoders_newname =  decoders

time_df_forward = pd.DataFrame.from_records(decoders_forward, columns=configs_numbers)
time_df_forward["index"] =  decoders_newname
time_df_forward.set_index("index", inplace=True)
print(time_df_forward)

time_df_backwards = pd.DataFrame.from_records(decoders_backwards, columns=configs_numbers)
time_df_backwards["index"] =  decoders_newname
time_df_backwards.set_index("index", inplace=True)
print(time_df_backwards)

time_df_both = pd.DataFrame.from_records(decoders_both, columns=configs_numbers)
time_df_both["index"] =  decoders_newname
time_df_both.set_index("index", inplace=True)
print(time_df_both)



table_file = "execution_time_forward.tex"
results_file = latex_dir / table_file
with open(results_file, 'w') as tf:
     res = time_df_forward.to_latex(index_names = False)
     print(res)
     tf.write(res)

table_file = "execution_time_backwards.tex"
results_file = latex_dir / table_file
with open(results_file, 'w') as tf:
     res = time_df_backwards.to_latex(index_names = False)
     print(res)
     tf.write(res)    

table_file = "execution_time_both.tex"
results_file = latex_dir / table_file
with open(results_file, 'w') as tf:
     res = time_df_both.to_latex(index_names = False)
     print(res)
     tf.write(res)  


###########################
if False:
    decoders_others = ['LogNormMix', 'Exponential', 'RMTPP']
    config = "execution_time"

    res_forward, res_backwards, res_both = [], [], []
    for decoder in decoders_others: 
            suffix = str(dataset).replace("/", "-") + "-" + str(decoder) + '-' +  str(config) + '-' + str(run)
            execution_time_file = res_dir / (suffix + 'execution_time.pt')
            loaded = torch.load(execution_time_file, map_location=torch.device('cpu'))

            res = [ (b-a, c-b) for a, b, c in zip(loaded["start_time_forward"], loaded["start_time_backward"], loaded["end_time_backward"])]

            forward_times = [b-a for a, b in zip(loaded["start_time_forward"], loaded["start_time_backward"])]
            backward_times = [c-b for b, c in zip(loaded["start_time_backward"], loaded["end_time_backward"])]

            print(np.mean(forward_times), np.std(forward_times))
            print(np.mean(backward_times), np.std(backward_times))

            x_forward = "{:0.3f}".format(np.mean(forward_times)) + " (" + "{:0.3f}".format(np.std(forward_times)) + ")" 
            x_backwards = "{:0.3f}".format(np.mean(backward_times)) + " (" + "{:0.3f}".format(np.std(backward_times)) + ")"

            sum_times = forward_times + backward_times 
            x_both = "{:0.3f}".format(np.mean(sum_times)) + " (" + "{:0.3f}".format(np.std(sum_times)) + ")"

            res_forward.append(x_forward)
            res_backwards.append(x_backwards)
            res_both.append(x_both)

if False: 

    time_df_forward = pd.DataFrame.from_records(config_forward)
    time_df_forward["index"] =  decoders_others
    time_df_forward.set_index("index", inplace=True)
    print(time_df_forward)
    breakpoint()

    table_file = "execution_time_forward_others.tex"
    results_file = latex_dir / table_file
    with open(results_file, 'w') as tf:
        res = config_forward.to_latex(index_names = False)
        print(res)
        tf.write(res)

    table_file = "execution_time_backwards_others.tex"
    results_file = latex_dir / table_file
    with open(results_file, 'w') as tf:
        res = config_backwards.to_latex(index_names = False)
        print(res)
        tf.write(res)  

