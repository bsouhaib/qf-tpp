from typing import MappingView
import numpy as np
import torch

def display_metrics_name(metrics_name):
    if metrics_name == 'crps':
        return "QSm"
    elif metrics_name == 'logs':
        return "NLL"
    elif metrics_name == "mace":
        return "MACE"
    elif metrics_name == "is90":
        return "IS90"
    elif metrics_name == "piw90":
        return "IW90"
    elif metrics_name == "is50":
        return "IS50"
    elif metrics_name == "piw50":
        return "IW50"
    elif metrics_name == "qs50":
        return "QS50"
    elif metrics_name == "qs90":
        return "QS90"
    elif metrics_name == "smape":
        return "SMAPE"
    else:
        return metrics_name

def display_dataset_name(data_name, add_hline = False):
    if data_name == 'synth/poisson':
        new_name = "Poisson"
    elif data_name == 'synth/renewal':
        new_name = "Renewal"
    elif data_name == "synth/self_correcting":
        new_name = "Self-correcting"
    elif data_name == "synth/hawkes1":
        new_name = "Hawkes1"
    elif data_name == "synth/hawkes2":
        new_name = "Hawkes2"
    elif data_name == "taxi":
        new_name = "Taxi"
    elif data_name == "reddit_askscience_comments":
        new_name = "Reddit-C"
    elif data_name == "yelp_toronto":
        new_name = "Yelp"
    elif data_name == "wikipedia":
        new_name = "Wikipedia"
    elif data_name == "mooc":
        new_name = "MOOC"
    elif data_name == "lastfm":
        new_name = "LastFM"
    elif data_name == "twitter":
        new_name = "Twitter"
    elif data_name == "pubg":
        new_name = "PUBG"
    elif data_name == "reddit":
        new_name = "Reddit"
    elif data_name == "yelp_airport":
        new_name = "Yelp A"
    elif data_name == "yelp_mississauga":
        new_name = "Yelp M"
    elif data_name == "reddit_politics_submissions":
        new_name = "Reddit-S"
    else:
        new_name = data_name
        #raise("Error in name!")
    if add_hline:
        return "\\hline " + new_name
    else:
        return new_name

def display_decoder_name(decoder_name):
    if decoder_name == 'RQS_EXP-crps_qapprox':
        return "RQS-QF"
    if decoder_name == 'RQS_EXP-crps':
        return "RQS-QF-CRPS"
    elif decoder_name == 'RQS_EXP-crps_qapprox_100':
        return "RQS-QF-CRPS-100"
    elif decoder_name == 'RQS_EXP-crps_qapprox_200':
        return "RQS-QF-CRPS-200"
    elif decoder_name == 'RQS_EXP-crps_qapprox_500':
        return "RQS-QF-CRPS-500"
    elif decoder_name == 'RQS_EXP-crps_approx_quantile_reg':
        return "RQS_EXP-crps"
    else:
        return decoder_name
    

def coef_variation(list_pred):
    all_coef_var = []
    for item in list_pred:
        samples = item[1]
        all_coef_var.append(torch.mean(torch.std(samples, -1)/torch.mean(samples, -1)).item())
        
    return all_coef_var

def crps_sampling(list_pred):
    all_crps = []
    for item in list_pred:
        y = item[0]
        samples = item[1]

        N = samples.shape[-1]
        sample_sorted, _  = torch.sort(samples, dim = 2)
        indices = torch.arange(1, N+1)
        check_y = y.unsqueeze(-1) < sample_sorted
        qs = (2.0/N) * (sample_sorted - y.unsqueeze(-1)) * (N * check_y - indices.unsqueeze(0) + 0.5)
        crps = torch.mean(qs, dim = 2)
        all_crps.append(torch.mean(crps).item())
    
    return np.mean(all_crps)

def metrics(list_pred, analytical = False, mace_only = False):
    all_ace_q, all_picp, all_interval_scores, all_mace_pi, all_pi_lengths, all_ace_pi  = [], [], [], [], [], []
    all_pits, all_qs, all_crps, all_se, all_ae, all_ape, all_spe, all_coefvar = [], [], [], [], [], [], [], []
    all_sape = []
    for item in list_pred:
        y = item[0]

        if analytical:
            qf = item[2]

            n_quantiles = qf.shape[-1]

            n = n_quantiles + 1
            probs = torch.arange(1, n)/(n)

            pits, qs, crps, se, ae, ape = [], [], [], [], []

        else:
            samples = item[1]

            n_quantiles = 99 # 199
            n =  n_quantiles + 1 # 100 # 200
            probs = torch.arange(1,n)/n # n - 1 quantiles

            qf = torch.quantile(samples, probs, dim = 2).permute([1, 2, 0])

            if not mace_only:
                #
                N = samples.shape[2] 
                sample_sorted, _  = torch.sort(samples, dim = 2)
                indices = torch.arange(1, N+1)
                check_y = y.unsqueeze(-1) < sample_sorted


                # PITS
                pits = (torch.sum(~check_y, dim=-1) + 1)/(N+1)
                all_pits.append(pits)

                # Quantile scores
                qs = (2.0/N) * (sample_sorted - y.unsqueeze(-1)) * (N * check_y - indices.unsqueeze(0) + 0.5)
                all_qs.append(qs)


                # CRPS
                crps = torch.mean(qs, dim = 2)
                all_crps.append(crps)

                pointf = qf[:, :, probs == 0.5].squeeze(-1)
                #pointf =  = torch.mean(samples, -1)

                # SE
                se = (pointf - y).pow(2)
                all_se.append(se)

                # APE
                ape = torch.abs( (pointf - y)/y )
                all_ape.append(ape)

                # SAPE
                sape = 2 * ( torch.abs(pointf - y) / (torch.abs(pointf) + torch.abs(y)) )
                all_sape.append(sape)

                # SPE
                spe = torch.pow( (pointf - y)/y, 2)
                all_spe.append(spe)

                # AE
                #medianf = torch.median(samples, -1)[0] # Available through qs
                ae = torch.abs(pointf - y)
                all_ae.append(ae)

                # Coefficient of variation
                all_coefvar.append( torch.std(samples, -1)/torch.mean(samples, -1) )    

        # ACE
        ace_q = torch.abs( probs.unsqueeze(0) - torch.sum(y.unsqueeze(-1) < qf, axis = 1)/ y.shape[1])
        all_ace_q.append(ace_q)

        if not mace_only:
            # 
            c = int(n/2)
            target_interval_coverage = (probs[np.arange(c, n-1)] - probs[np.arange(c-2, -1, -1)])

            interval_scores_list = []
            pi_coverage_indicators_list =  []
            pi_lengths_list = []
            for k in np.arange(1, c):
                i_l = (c-1) - k 
                i_u = (c-1) + k

                q_L = qf[:, :, i_l].squeeze(-1) 
                q_U = qf[:, :, i_u].squeeze(-1)

                # Interval Score
                alpha = 1 - (probs[i_u] - probs[i_l])
                interval_scores_k = (q_U - q_L)  + (2.0 / alpha) * (q_L - y) * (y <= q_L)  + (2.0 / alpha) * (y - q_U ) * (y >= q_U)             
                interval_scores_list.append(interval_scores_k)

                # 
                pi_lengths_list.append(q_U - q_L)

                # PICP
                pi_coverage_indicators_k = ((y >= q_L) *  (y <= q_U))
                pi_coverage_indicators_list.append(pi_coverage_indicators_k)

            picp_i = torch.sum(torch.stack(pi_coverage_indicators_list, 2), 1 )/y.shape[1]
            all_picp.append(picp_i)

            pi_length_i = torch.sum(torch.stack(pi_lengths_list, 2), 1 )/y.shape[1]
            all_pi_lengths.append(pi_length_i)

            all_interval_scores.append(torch.sum(torch.stack(interval_scores_list, 2), 1 )/y.shape[1])
            
            ace_pi = [np.abs(tic - picp).item() for tic, picp in zip(target_interval_coverage, picp_i.squeeze().tolist())]
            all_ace_pi.append(ace_pi)

            mean_ace_pi =  np.mean([np.abs(tic - picp) for tic, picp in zip(target_interval_coverage, picp_i.squeeze().tolist())])        
            all_mace_pi.append(mean_ace_pi)

    ace = torch.mean(torch.stack(all_ace_q, 2), -1).squeeze() # averaged over all test sequences
    mace = torch.mean(ace).item()

    if not mace_only:
        picp = torch.mean(torch.stack(all_picp, 2), -1).squeeze()
        int_scores =  torch.mean(torch.stack(all_interval_scores, 2), -1).squeeze()
        mace_pi = np.mean(all_mace_pi).item() 

        # 
        is10 = int_scores.squeeze()[4].item()
        is50 = int_scores.squeeze()[24].item()
        is90 = int_scores.squeeze()[44].item()

        pilen = torch.mean(torch.stack(all_pi_lengths, 2), -1).squeeze()

        piw10 = pilen.squeeze()[4].item()
        piw50 = pilen.squeeze()[24].item()
        piw90 = pilen.squeeze()[44].item()


        ace_pi = torch.tensor(np.mean(np.stack(all_ace_pi, 1), -1))

        pits = torch.cat(all_pits, 1).squeeze()

        qs = torch.mean(torch.cat(all_qs, 1), (0, 1))

        crps = torch.mean(torch.cat(all_crps, 1)).item()

        mape = torch.mean(torch.cat(all_ape, 1)).item()

        smape = torch.mean(torch.cat(all_sape, 1)).item()

        mspe = torch.mean(torch.cat(all_spe, 1)).item()

        mse = torch.mean(torch.cat(all_se, 1)).item()
        mae = torch.mean(torch.cat(all_ae, 1)).item()

        mcv = torch.mean(torch.cat(all_coefvar, 1)).item()

    
    # ace_pi 

    if not mace_only:
        return {"ace": ace, 
            "picp": picp, 
            "int_scores": int_scores, 
            "mace": mace, 
            "pilen": pilen, 
            "ace_pi": ace_pi,
            ####
            #"pits": pits,
            "qs": qs,
            ####
            "crps": crps,
            "mape": mape,
            "smape": smape,
            "mspe": mspe,
            "mse": mse,
            "mae": mae,
            "mcv": mcv,
            ###
            "is10": is10,
            "is50": is50,
            "is90": is90,
            ###
            "piw10": piw10,
            "piw50": piw50,
            "piw90": piw90
            } 
    else:
        return {"mace": mace}


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict

def dict_ste(dict_list):
    ste_dict = {}
    for key in dict_list[0].keys():
        if torch.is_tensor(dict_list[0][key]):
            ste_dict[key] = torch.stack([d[key].squeeze() for d in dict_list]).std(0) / np.sqrt(len(dict_list))
        else:
            ste_dict[key] = np.std([d[key] for d in dict_list]) / np.sqrt(len(dict_list))
    return ste_dict

def dict_mean_ste(dict_list):
    mean_dict = dict()
    ste_dict = dict()
    mean_ste_dict = dict()
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)

        if torch.is_tensor(dict_list[0][key]):
            ste_dict[key] = torch.stack([d[key].squeeze() for d in dict_list]).std(0) / np.sqrt(len(dict_list))
            mean_ste_dict[key] = ["{:0.3f}".format(m) + " (" + "{:0.3f}".format(s) + ")" for m, s in zip(mean_dict[key].squeeze().tolist(), ste_dict[key].tolist())]
        else:
            ste_dict[key] = np.std([d[key] for d in dict_list]) / np.sqrt(len(dict_list))
            
            mean_ste_dict[key] = "{:0.3f}".format(mean_dict[key]) + " (" + "{:0.3f}".format(ste_dict[key]) + ")"

    return mean_ste_dict