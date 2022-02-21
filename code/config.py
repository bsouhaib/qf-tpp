

def update_args(args):

    # Default
    args.batch_size_train = 64 # 128
    args.learning_rate = 1e-3   
    
    res_split = args.method.split("-")
    args.decoder_name = res_split[0]
    args.scoring_rule = res_split[1] if len(res_split) == 2 else "logs"

    args.N_crps_approx = 100
    args.N_calibration = 100

    if "crps_qapprox_100" in args.scoring_rule:
        args.N_crps_approx = 100
        args.scoring_rule = "crps_qapprox"
    elif "crps_qapprox_200" in args.scoring_rule:
        args.N_crps_approx = 200
        args.scoring_rule = "crps_qapprox"
    elif "crps_qapprox_500" in args.scoring_rule:
        args.N_crps_approx = 500
        args.scoring_rule = "crps_qapprox"

    #
    if "crps" in args.scoring_rule:
        args.max_epochs = 700 #300
        args.patience = 50
        args.display_step = 1
    elif args.scoring_rule == "logs":
        args.max_epochs = 2000
        args.patience = 100
        args.display_step = 1 #10

    args.right = 0.95 #0.99 #0.95 #0.99
    args.train_widths =  True

    args.log_and_scaling = True

    args.execution_time = False

    ########
    args.threshold_loss_val = 1e-4
    if "crps" in args.scoring_rule:
        if args.dataset_name == 'reddit_politics_submissions':
            args.threshold_loss_val = 1e-5

    print(args.config)

    ########
    if args.decoder_name == "RQS_EXP": 
        if args.config == "1":
            args.num_bins = 1
        elif args.config == "2":
            args.num_bins = 2
        elif args.config == "3":
            args.num_bins = 3
        elif args.config == "4":
            args.num_bins = 4
        elif args.config == "5":
            args.num_bins = 5
        elif args.config == "6":
            args.num_bins = 6
        elif args.config == "7":
            args.num_bins = 7
        elif args.config == "8":
            args.num_bins = 8
        elif args.config == "10":
            args.num_bins = 10
            args.regularization = 0
        elif args.config == "10b":
            args.num_bins = 10
            args.regularization = 1e-5 
        elif args.config == "10c":
            args.num_bins = 10
            args.regularization = 1e-3 
        elif args.config == "10d":
            args.num_bins = 10
            args.regularization = 1e-1
        elif args.config == "15":
            args.num_bins = 15
            args.regularization = 0
        elif args.config == "20":
            args.num_bins = 20
            args.regularization = 0
        elif args.config == "test":
            args.num_bins = 10
        elif args.config == "1_time":
            args.num_bins = 1
            args.execution_time = True
            args.max_epochs = 10
            args.log_and_scaling = False
        elif args.config == "2_time":
            args.num_bins = 2
            args.execution_time = True
            args.max_epochs = 10
            args.log_and_scaling = False
        elif args.config == "3_time":
            args.num_bins = 3
            args.execution_time = True
            args.max_epochs = 10
            args.log_and_scaling = False
        elif args.config == "4_time":
            args.num_bins = 4
            args.execution_time = True
            args.max_epochs = 10
            args.log_and_scaling = False
        elif args.config == "5_time":
            args.num_bins = 5
            args.execution_time = True
            args.max_epochs = 10
            args.log_and_scaling = False
        elif args.config == "6_time":
            args.num_bins = 6
            args.execution_time = True
            args.max_epochs = 10
            args.log_and_scaling = False
        elif args.config == "7_time":
            args.num_bins = 7
            args.execution_time = True
            args.max_epochs = 10
            args.log_and_scaling = False
        elif args.config == "8_time":
            args.num_bins = 8
            args.execution_time = True
            args.max_epochs = 10
            args.log_and_scaling = False
        elif args.config == "10_time":
            args.num_bins = 10
            args.execution_time = True
            args.max_epochs = 10
            args.log_and_scaling = False
        elif args.config == "15_time":
            args.num_bins = 15
            args.execution_time = True
            args.max_epochs = 10
            args.log_and_scaling = False
        else:
            raise("This config. does not exist!")
    else:
        if args.config == "execution_time":
            args.execution_time = True
            args.max_epochs = 10
            args.n_components = 64
            args.regularization = 0   
        elif args.config == "64":
            args.n_components = 64
            args.regularization = 0  
        elif args.config == "64b":
            args.n_components = 64
            args.regularization = 1e-5
        elif args.config == "64c":
            args.n_components = 64
            args.regularization = 1e-3
        else:
            raise("This config. does not exist!")

    # LogNormMix generates very large values (overflow) 
    if args.dataset_name == "lastfm" or args.dataset_name == "reddit":
        args.n_components = 8
        print("args.n_components ---> ", args.n_components)
    
    return args 

def print_args(args):

    print("*********")
    print("*********")
    print(" dataset_name: ", args.dataset_name, "\n",
            "method: ", args.method, "\n", 
            "decoder_name: ", args.decoder_name, "\n", 
            "scoring_rule: ", args.scoring_rule, "\n", 
            "config: ", args.config, "\n", 
            "run: ", args.run, "\n", "\n", 

            "max_epochs: ", args.max_epochs, "\n", 
            "patience: ", args.patience, "\n", 
            "threshold_loss_val: ", args.threshold_loss_val, "\n", 
            "batch_size_train: ", args.batch_size_train, "\n", 
            "learning_rate: ", args.learning_rate, "\n", 
            "regularization: ", args.regularization, "\n", "\n", 

            "train_widths: ", args.train_widths, "\n", 
            "num_bins: ", args.num_bins, "\n", 
            "right: ", args.right, "\n", 
            "log_and_scaling: ", args.log_and_scaling, "\n", 
            "calib_regularization: ", args.calib_regularization, "\n", 
            "N_calibration: ", args.N_calibration, "\n", 
            "N_crps_approx: ", args.N_crps_approx, "\n", "\n", 

            "split: ", args.split,  " - ",
            "seq_len", args.seq_len,  " - ",
            "use_history: ", args.use_history, " - ",
            "use_embedding: ", args.use_embedding, " - ",

            "rnn_type: ", args.rnn_type, " - ",
            "embedding_size: ", args.embedding_size, " - ",
            "history_size: ", args.history_size, " - ",

            "hypernet_hidden_sizes: ", args.hypernet_hidden_sizes, " - ",
            "layer_size: ", args.layer_size, " - ",

            "n_components: ", args.n_components, " - ",   
            "max_degree: ", args.max_degree, " - ", 
            "n_layers: ", args.n_layers, " - ", 
            "n_terms: ", args.n_terms,  " - ",
            "trainable_affine: ", args.trainable_affine, " - ",

            "display_step: ", args.display_step, " - ",
            "execution_time: ", args.execution_time, " - ",
            "num_workers: ", args.num_workers, " - ",
            "batch_size_val: ", args.batch_size_val, " - ",
            "batch_size_test: ", args.batch_size_test)
    print("*********")
    print("*********")
