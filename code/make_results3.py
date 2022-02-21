
#import tensorflow as tf
import tensorflow 
from tensorflow.python.summary.summary_iterator import summary_iterator
import glob
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import tensorboard as tb




#datasets = ['wikipedia', 'yelp_toronto', 'reddit_askscience_comments']
#decoders = ['LogNormMix', 'RQS_EXP-logs', 'RQS_EXP-crps_qapprox', 'RMTPP', 'Exponential']

datasets = ['lastfm']

####
#decoders = ['RQS_EXP/crps_qapprox_reg']
#decoders = ['RQS_EXP/crps_qapprox']
#decoders = ['LogNormMix']
decoders = ['Exponential']

do_calib_only = False
color_decoders_loss = ["red", "red"]
color_decoders_calib = ["blue", "blue"]


#configs = ["config1", "config2", "config3"]
configs = ["64"]
run = "1"

pdfs_dir  = Path("../work/pdfs/")
pdfs_dir.mkdir(parents=True, exist_ok=True)


if False:
    dataset = "reddit"
    decoder='RQS_EXP/crps_qapprox'
    config = "config3"
    experiment_id  = Path("../work/tensorboard/") / dataset / decoder / config /run /"events.out.tfevents.1631893417.Souhaibs-MacBook-Pro.local.43398.0"
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()
    df


all_log = []
for dataset in datasets:
    fig, ax1 = plt.subplots()
    for i_decoder, decoder in enumerate(decoders):
        #fig = plt.figure()

        for config in configs:
            my_files = glob.glob(os.path.join("../work/tensorboard/", dataset, decoder, config, run, "event*"))
                    
            assert(len(my_files) > 0)
            
            print(my_files)
            if len(my_files) > 1:
                print("THERE ARE MULTIPLE FILES !!!!")

            #file_log = my_files[0]
            myfile = "events.out.tfevents.1634062914.ip-172-31-44-72.71817.0"
            file_log = '../work/tensorboard/lastfm/Exponential/64/1/' +  myfile

            loss_val = []
            calib_val = []
            for e in summary_iterator(file_log):
                for v in e.summary.value:
                    #r = {'metric': v.tag, 'value':v.simple_value}
                    if v.tag == "Loss/val_loss_logs":
                        loss_val.append(v.simple_value)
                    elif v.tag == "Loss/val_calib":
                        calib_val.append(v.simple_value)
                    #print(loss_val)
   
            #all_log.append(runlog)
            #if "crps" in decoder:
            #   #loss_val = np.log(loss_val)
            #    pass
            #breakpoint()
            if not do_calib_only:
                ax1.plot(loss_val, label = config)
                ax1.plot(loss_val,  "o", markersize= 3, color = color_decoders_loss[i_decoder])
                ax1.set_xlabel("Epoch")
                
                if "crps" in decoder:
                    ax1.set_ylabel("CRPS")
                else:
                    ax1.set_ylabel("NLL")

            ax2 = ax1.twinx()
            ax2.plot(calib_val)
            ax2.plot(calib_val,  "o", markersize= 2, color = color_decoders_calib[i_decoder])
            ax2.tick_params(axis='y') #, labelcolor="red")
            ax2.set_ylabel("Absolute Calibration Error")


        plt.legend()
    plt.show()
    fig.savefig(pdfs_dir / ("curves_" + dataset + "_" + decoder.replace("/", "-") + ".pdf") , bbox_inches='tight')

breakpoint()