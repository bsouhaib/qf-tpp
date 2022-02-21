import argparse

def myparser():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--run', type=int,  default = 1, help = 'run id')

    parser.add_argument('--dataset_name', metavar='dataset', type=str,
                        help= 'The dataset to use.',
                        default = 'reddit' ,
                        choices = ["taxi", "yelp_airport", "yelp_mississauga", 'pubg', 'reddit_askscience_comments', 'reddit_politics_submissions', 'twitter', 
                        'yelp_toronto', 'wikipedia', 'mooc', 'stack_overflow', 'lastfm', 'reddit', 
                        'synth/poisson', 'synth/renewal', 'synth/self_correcting', 'synth/hawkes1', 'synth/hawkes2', 'wikipedia_new', 'reddit_new', 'yelp_toronto_new'])
                                        
    parser.add_argument('--split', type=str, default = 'whole_sequences',  help = 'How to split the sequences (each_sequence -- split every seq. into train/val/test)')
    parser.add_argument('--use_history', type=bool, default = True,  help = 'Whether to use RNN to encode history')
    parser.add_argument('--history_size', type=int, default = 64,  help = 'Size of the RNN hidden vector')
    parser.add_argument('--rnn_type', type=str, default = 'RNN',  help = 'Which RNN cell to use (other: [GRU, LSTM])', choices=['GRU', 'LSTM'])
    parser.add_argument('--use_embedding', type=bool, default = False,  help = 'Whether to use sequence embedding (should use with each_sequence split)')
    
    ## General model config
    parser.add_argument('--embedding_size', type=int, default = 32,  help = 'Size of the sequence embedding vector. IMPORTANT: when using split = whole_sequences, the model will only learn embeddings for the training sequences, and not for validation / test')
    parser.add_argument('--trainable_affine', type=bool, default = False,  help = 'Train the final affine layer?')

    ## Decoder config
    methods = ['RMTPP', 'FullyNeuralNet', 'Exponential', 'LogNormMix']
    methods = methods + ['RQS_EXP-logs', 'RQS_EXP-crps', 'RQS_EXP-crps_qapprox', 'RQS_EXP-crps_qapprox_reg', 'RQS_EXP-crps_qapprox_100', 'RQS_EXP-crps_qapprox_500', 'RQS_EXP-crps_qapprox_200']
    #'EXP', 'RQS_PTO'
    parser.add_argument('--method', type=str,  default = 'RQS_EXP-crps_qapprox', help = 'Method', choices = methods)
    parser.add_argument('--n_components', type=int,  default = 16, help = ' Number of components for a mixture model')
    parser.add_argument('--hypernet_hidden_sizes', type=list,  default = [], help = 'Number of units in MLP generating parameters ([] -- affine layer, [64] -- one layer, etc.)')

    ## Flow params
    # Polynomial
    parser.add_argument('--max_degree', type=int,  default = 3, help = 'Maximum degree value for Sum-of-squares polynomial flow (SOS)')
    parser.add_argument('--n_terms', type=int,  default = 4, help = 'Number of terms for SOS flow')

    # DSF / FullyNN
    parser.add_argument('--n_layers', type=int,  default = 2, help = 'Number of layers for Deep Sigmoidal Flow (DSF) / Fully Neural Network flow (Omi et al., 2019)')
    parser.add_argument('--layer_size', type=int,  default = 64, help = 'Number of mixture components / units in a layer for DSF and FullyNN')

    ## Training config
    parser.add_argument('--scoring_rule', type=str,  default = 'logs', help = 'Scoring rule', choices = ['logs', 'crps', 'crps_qapprox'])
    parser.add_argument('--regularization', type=float,  default = 0, help = 'L2 regularization parameter')
    parser.add_argument('--learning_rate', type=float,  default = 1e-3, help = 'Learning rate for Adam optimizer')

    parser.add_argument('--max_epochs', type=int,  default = 1000, help = 'For how many epochs to train')
    parser.add_argument('--display_step', type=int,  default = 1, help = 'Display training statistics after every display_step')
    parser.add_argument('--patience', type=int,  default = 100, help = 'After how many consecutive epochs without improvement of val loss to stop training')

    parser.add_argument('--calib_regularization', type=float,  default = 0, help = 'Calibration regularization')

    ##  ADDED BY SBT
    parser.add_argument('--seq_len', type=int,  default = 128, help = 'Sequence lengths')
    parser.add_argument('--batch_size_train', type=int,  default = 64, help = 'Batch size train')
    parser.add_argument('--batch_size_val', type=int,  default = 1, help = 'Batch size val')
    parser.add_argument('--batch_size_test', type=int,  default = 1, help = 'Batch size test')

    ## RQS
    parser.add_argument('--num_bins', type=int,  default = 5, help = 'Number of bins')
    parser.add_argument('--right', type=float,  default = 0.95, help = 'Upper tail threshold probability')
    parser.add_argument('--train_widths', type=bool,  default = False, help = 'train_widths')
    parser.add_argument('--log_and_scaling', type=bool,  default = False, help = 'log_and_scaling')


    parser.add_argument('--config', type=str,  default = "5", help = 'config')

    parser.add_argument('--num_workers', type=int,  default = 0, help = 'num_workers')
    parser.add_argument('--num_threads', type=int,  default = 1, help = 'num_threads')



    return parser