import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from scipy.stats import expon
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


dataset = "synth/poisson"
if dataset == "synth/poisson":
        plot_true = True

do_pdf = False

if False:
    #decoder = "LogNormMix"
    decoder = "Exponential"
    config = "64"
else:
    decoder = "RQS_EXP-crps_qapprox"
    config = "3"
    do_scaling = True


run = 1

res_dir = Path("../work/results/")
pdfs_dir  = Path("../work/pdfs/")

suffix = str(dataset).replace("/", "-") + "-" + str(decoder)  + '-' +  str(config) + '-'  + str(run) + '.pt'
results_file = res_dir / suffix

loaded = torch.load(results_file, map_location=torch.device('cpu'))

id_sequence = 0 
id_test = 10
quantities = loaded["predictions_test"][id_sequence] 
# (t, samples, quantiles, q_densities, params) 
scale_init = loaded["std_out_train"]

true_values, samples, quantiles, q_densities, params = quantities




##
cumwidths = params[0][0][0, id_test, :].detach()
cumheights = params[0][2][0, id_test, :].detach()
derivatives = params[0][4][0, id_test, :].detach()

normalized_params_tail = params[1][0][0, id_test, :].detach()
top = params[2][0][id_test, :].detach()

n_quantiles = 99
probs = np.arange(1, n_quantiles+1)/(n_quantiles + 1)


#print("REMOVE BIS !!!!")
#probs[19] = 0.1980
#probs[96] = 0.992
#probs[97] = 0.995
#probs[98] = 0.997

qtile = quantiles[0, id_test, :].detach()
samp = samples[0, id_test, :].detach()
d_q = q_densities[0, id_test, :].detach()
#d_s = s_densities[0, id_test, :].detach()

myfile = pdfs_dir / ('toy_example_' + dataset.replace("/", "-") +'.pdf')
with PdfPages(myfile) as pdf:

    if do_pdf:
        fig, ax = plt.subplots(1, 2, figsize=(8.27, 3))    # 8.3 x 11.7 inches
    else:
        fig, ax = plt.subplots(1, 1, figsize=(4, 2))  


    if do_pdf:
        ax[0].plot(probs, qtile, "o",  label = "RQS")
        ax[0].set_xlabel("Probability level")
        ax[0].set_ylabel("Inter-arrival times")
    else:
        ax.plot(probs, qtile, label = "RQS-QF", color = "grey")
        ax.set_xlabel("Probability level")
        ax.set_ylabel("Inter-arrival times")
    
    #ax[0][0].plot(cumwidths, cumheights, "o", markersize= 3, color = "orange")
    if not do_pdf:
        if do_scaling:
            ax.plot(cumwidths, torch.exp(cumheights * scale_init) - 1.0, "o", markersize= 3, color = "orange", label = "Knots")
        else:
            ax.plot(cumwidths, cumheights, "o", markersize= 3, color = "orange", label = "Knots")
    else:
        if do_scaling:
            ax[0].plot(cumwidths, torch.exp(cumheights * scale_init) - 1.0, "o", markersize= 3, color = "orange", label = "Knots")
        else:
            ax[0].plot(cumwidths, cumheights, "o", markersize= 3, color = "orange", label = "Knots")


    if plot_true:
        idx = np.arange(0, 95, 3).tolist() + np.arange(95, 99).tolist()
        probs_subset = probs[idx]
        z = [expon.ppf(x) for x in probs_subset]
        ax.plot(probs_subset, z, "o", color='black',  markersize= 1,  label = "True quantile function") # linestyle='dashed',
        ax.legend(fontsize = "x-small")

    if False:
        ax[1].plot(qtile, probs, label = "RQS", color = "grey")

        if do_scaling:
            ax[1].plot(torch.exp(cumheights * scale_init) - 1.0, cumwidths, "o", markersize= 3, color = "orange", label = "Knots")
        else:
            ax[1].plot(cumheights, cumwidths, "o", markersize= 6, color = "orange", label = "Knots")

        ax[1].plot(z, probs, "o", color='grey',  markersize= 1,  label = "True CDF") # linestyle='dashed',
        ax[1].set_xlabel("Inter-arrival times")
        ax[1].set_ylabel("Probability level")
        ax[1].legend(fontsize = "x-small")


    if do_pdf:
        ax[1].plot(qtile, d_q, label = "RQS")
        ax[1].set_xlabel("Inter-arrival times")
        ax[1].set_ylabel("Density")

        if plot_true:
            z = [expon.pdf(x) for x in qtile]
            ax[1].plot(qtile, z, "o", color='grey',label = "True PDF",  markersize= 1)

        if do_scaling:
            x = torch.exp(cumheights * scale_init) - 1.0
            ax[1].plot(x,  1/ (derivatives * torch.exp(cumheights * scale_init) * scale_init ), "o", markersize= 3, color = "orange", label = "Knots")
        else:
            ax[1].plot(cumheights, 1/derivatives, "o", markersize= 3, color = "orange", label = "Knots")

    if not do_pdf:
        ax.legend(fontsize = "x-small")


    fig.tight_layout()

    pdf.savefig()
    plt.close()

breakpoint()

if False:
    true_values = true_values.squeeze()
    samples = samples.squeeze()

    myfile = pdfs_dir / ('viz_' + dataset.replace("/", "") + '.pdf')
    with PdfPages(myfile) as pdf:

        plt.figure()

        plt.hist(samples[0, :].tolist(), density = True)

        allx  = np.arange(0, 20, 0.1)
        ally = [expon.pdf(x) for x in allx]
        plt.plot(allx, ally)
        plt.axvline(x=1, color = "red")

        pdf.savefig()
        plt.close()


    breakpoint()