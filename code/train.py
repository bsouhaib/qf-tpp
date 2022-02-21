import dpp
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

import pprint
import time
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from make_parser import myparser
from config import update_args, print_args

#torch.set_default_tensor_type(torch.cuda.FloatTensor) # torch.cuda.is_available()
torch.set_default_tensor_type(torch.FloatTensor)

my_parser = myparser()
args = my_parser.parse_args()

args = update_args(args)

# Number of threads (multi-core environment)
print("\n ***** HYPERTHREADING ****")
print("Number of threads available: ", torch.get_num_threads())
torch.set_num_threads(args.num_threads)
print("Number of threads used: ", torch.get_num_threads())
print("************************* \n")


# Different seed for each run
seed = args.run
np.random.seed(seed)
torch.manual_seed(seed)

suffix = str(args.dataset_name).replace("/", "-") + "-" + str(args.method) + '-' +  str(args.config) + '-' + str(args.run)

# Writer
tensorboard_folder =  Path("../work/tensorboard/")
tensorboard_dir = tensorboard_folder / suffix.replace("-", "/")
tensorboard_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(tensorboard_dir, flush_secs = 10)

res_dir = Path("../work/results/")
log_dir = Path("../work/out/")
res_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)
results_file = res_dir / (suffix + '.pt')
execution_time_file = res_dir / (suffix + 'execution_time.pt')

#using_script = len(sys.argv) > 1
using_script =  False
if using_script:
    log_file = log_dir / (suffix + '.txt')
    sys.stdout = open(log_file, "w")

### Data
print('Loading data...')
if '+' not in args.dataset_name:
    dataset = dpp.data.load_dataset(args.dataset_name)
else:
    # If '+' in dataset_name, load all the datasets together and concatenate them
    # For example, dataset_name='synth/poisson+synth/renewal' loads poisson and renewal datasets
    dataset_names = [d.strip() for d in args.dataset_name.split('+')]
    dataset = dpp.data.load_dataset(dataset_names.pop(0))
    for d in dataset_names:
        dataset += dpp.data.load_dataset(dataset_names.pop(0))

###
print("-----")
print("Number of sequences: ", len(dataset))
print("Lengths of sequences (min, q_0.25, median, q_0.75, q_0.9, max): ", np.quantile([len(dataset[i][0]) for i in range(len(dataset))], [0, 0.25, 0.5, 0.75, 0.9, 1] ) )
print("-----")
#breakpoint()


# Split into train/val/test, on each sequence or assign whole sequences to different sets
if args.split == 'each_sequence':
    d_train, d_val, d_test = dataset.train_val_test_split_each(seed=seed)
elif args.split == 'whole_sequences':
    d_train, d_val, d_test = dataset.train_val_test_split_whole(seed=seed)
else:
    raise("Unsupported dataset split")
    #raise ValueError(f'Unsupported dataset split {args.split}')


# Calculate mean and std of the input inter-event times and normalize only input
mean_in_train, std_in_train = d_train.get_mean_std_in()
std_out_train = 1.0
d_train.normalize(mean_in_train, std_in_train, std_out_train)
d_val.normalize(mean_in_train, std_in_train, std_out_train)
d_test.normalize(mean_in_train, std_in_train, std_out_train)

# Break down long train sequences for faster batch traning and create torch DataLoaders
d_train.break_down_long_sequences(args.seq_len)
collate = dpp.data.collate
dl_train = torch.utils.data.DataLoader(d_train, batch_size=args.batch_size_train, shuffle=True, collate_fn=collate, num_workers = args.num_workers, pin_memory = False)
dl_val = torch.utils.data.DataLoader(d_val, batch_size=args.batch_size_val, shuffle=False, collate_fn=collate, num_workers = args.num_workers, pin_memory = False)
dl_test = torch.utils.data.DataLoader(d_test, batch_size=args.batch_size_test, shuffle=False, collate_fn=collate, num_workers = args.num_workers, pin_memory = False)

#
if args.decoder_name  in ["RQS_EXP", 'LogNormMix']:
    mean_out_train, std_out_train = d_train.get_log_mean_std_out()
elif args.decoder_name in ['RMTPP', 'FullyNeuralNet', 'Exponential', 'EXP']:
    _, std_out_train = d_train.get_mean_std_out()
    mean_out_train = 0.0
else:
    raise ValueError(f'Which transformation to apply with {args.decoder_name}?')


min_qtail, max_qtail = None, None
if args.decoder_name == 'RQS_EXP':
    # Mininmum and maximum alpha-quantile across sequences where alpha = args.right
    if args.log_and_scaling:
        #all_qtail = torch.stack([torch.quantile( torch.log(x)/std_out_train, args.right) for x in d_train.out_times])
        all_qtail = torch.stack([torch.quantile( torch.log(1.0 + x)/std_out_train, args.right) for x in d_train.out_times])
    else:
        all_qtail = torch.stack([torch.quantile(x, args.right) for x in d_train.out_times])
        
    min_qtail = all_qtail.min().item()
    max_qtail = all_qtail.max().item()

# Model setup
print('Building model...')

# General model config
general_config = dpp.model.ModelConfig(
    use_history=args.use_history,
    history_size=args.history_size,
    rnn_type=args.rnn_type,
    use_embedding=args.use_embedding,
    embedding_size=args.embedding_size,
    num_embeddings=len(dataset),
    )
 
# Decoder specific config
decoder = getattr(dpp.decoders, args.decoder_name)(general_config,
                                              n_components=args.n_components,
                                              hypernet_hidden_sizes=args.hypernet_hidden_sizes,

                                              max_degree=args.max_degree,
                                              n_terms=args.n_terms,
                                              n_layers=args.n_layers,
                                              layer_size=args.layer_size,

                                              shift_init=mean_out_train,
                                              scale_init=std_out_train,
                                              trainable_affine=args.trainable_affine,
                                              
                                              left = 0, 
                                              right = args.right, 
                                              bottom = 0, 
                                              
                                              min_qtail = min_qtail,
                                              max_qtail = max_qtail,


                                              num_bins = args.num_bins,
                                              train_widths = args.train_widths,
                                              log_and_scaling  = args.log_and_scaling,
                                              calib_regularization = args.calib_regularization
)

# Define model
model = dpp.model.Model(general_config, decoder)
model.use_history(general_config.use_history)
model.use_embedding(general_config.use_embedding)

# Define optimizer
opt = torch.optim.Adam(model.parameters(), weight_decay=args.regularization, lr=args.learning_rate)


# Traning
print('Starting training...', flush=True)
impatient = 0
best_loss = np.inf
best_model = deepcopy(model.state_dict())
training_val_losses = []

#
print_args(args)

start_time_forward, start_time_backward, end_time_backward = [], [], []
for epoch in range(args.max_epochs):
   
    model.train()
    for input in dl_train:
        opt.zero_grad()

        if args.execution_time: start_time_forward.append(time.time())
        score_batch = model.score(input, scoring_rule = args.scoring_rule, N = args.N_crps_approx)
        loss = model.aggregate(score_batch, input.length)
        if args.execution_time: start_time_backward.append(time.time())
        loss.backward()
        if args.execution_time: end_time_backward.append(time.time())
        opt.step()

    
    model.eval()

    # Validation error (crps/logs)
    loss_val = model.get_total_loss(dl_val, scoring_rule = args.scoring_rule, N = args.N_crps_approx)
    writer.add_scalar("Loss/val_loss_" + args.scoring_rule, loss_val, epoch)

    # Validation error (calibration)
    if args.decoder_name in ["LogNormMix", "RQS_EXP"]:
        calib_err_mean, calib_err_max = model.get_total_calibration_loss(dl_val, N = args.N_calibration)
        writer.add_scalar('Loss/val_calib', calib_err_mean, epoch)


    training_val_losses.append(loss_val.item())

    if (best_loss - loss_val) < args.threshold_loss_val:
        impatient += 1
        if loss_val < best_loss:
            best_loss = loss_val.item()
            best_model = deepcopy(model.state_dict())
    else:
        best_loss = loss_val.item()
        best_model = deepcopy(model.state_dict())
        impatient = 0

    if impatient >= args.patience:
        print(f'Breaking due to early stopping at epoch {epoch}')
        break

    if (epoch + 1) % args.display_step == 0:
        if args.decoder_name in ["LogNormMix", "RQS_EXP"]:
            print(f"Epoch {epoch+1:4d}, Pat. {impatient:4d}, loss_train_last_batch = {loss:.4f}, loss_val = {loss_val:.4f}, best_val = {best_loss:.4f}, calib_val_mean = {calib_err_mean:.4f}, calib_val_max = {calib_err_max:.4f}", flush=True)
        else:
            print(f"Epoch {epoch+1:4d}, Pat. {impatient:4d}, loss_train_last_batch = {loss:.4f}, loss_val = {loss_val:.4f}, best_val = {best_loss:.4f}", flush=True)

        print(datetime.now())

writer.close()

## Execution time
if args.execution_time:
    dict_exec_time = {"start_time_forward": start_time_forward, "start_time_backward":start_time_backward,  "end_time_backward": end_time_backward}
    torch.save(dict_exec_time, execution_time_file)
    print("Exited! ")
    sys.exit(0) # No need to compute the rest if I just want execution time

### Evaluation
model.load_state_dict(best_model)
model.eval()


print(datetime.now())
loss_train = model.get_total_loss(dl_train, scoring_rule=args.scoring_rule, N = args.N_crps_approx)
loss_val = model.get_total_loss(dl_val, scoring_rule=args.scoring_rule, N = args.N_crps_approx)
loss_test = model.get_total_loss(dl_test, scoring_rule=args.scoring_rule, N = args.N_crps_approx)
print(datetime.now())

print(f'Scores \n'
      f' - Train: {loss_train:.4f}\n'
      f' - Val:   {loss_val.item():.4f}\n'
      f' - Test:  {loss_test.item():.4f}')
print("-----")


# Save losses
dict_results = {"loss_train": loss_train, "loss_val":loss_val,  "loss_test": loss_test,  "args": args, "best_model": best_model}
torch.save(dict_results, results_file)

# Get and save test predictions (samples) 
print(datetime.now())
probs = None
if args.decoder_name == "RQS_EXP":
    n_quantiles = 99
    probs = np.arange(1, n_quantiles+1)/(n_quantiles + 1)

n_samples = 500 #1000 #500
predictions_test = model.get_predictions_and_true_obs(dl_test, n_samples, probs)

dict_results.update({"predictions_test": predictions_test})

dict_results.update({"mean_out_train": mean_out_train, "std_out_train": std_out_train})

torch.save(dict_results, results_file)

print("Done! ")

if using_script:
    sys.stdout.close()
