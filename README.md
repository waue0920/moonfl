# moonfl

# args

* ('--model', type=str, default='resnet50', help='neural network used intraining') # 載入base_model，定義於model.py的 ModelFedCon()
* ('--dataset', type=str, default='cifar100', help='dataset used for training')
* ('--net_config', type=lambda x: list(map(int, x.split(', '))))
* ('--partition', type=str, default='homo', help='the data partitioningstrategy')
* ('--batch-size', type=int, default=64, help='input batch size for training(default: 64)')
* ('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
* ('--epochs', type=int, default=5, help='number of local epochs')
* ('--n_parties', type=int, default=2, help='number of workers in a distributedcluster')
* ('--alg', type=str, default='fedavg', help='communication strategy: fedavg/fedprox')
* ('--comm_round', type=int, default=50, help='number of maximum communicationroun')
* ('--init_seed', type=int, default=0, help="Random seed")
* ('--dropout_p', type=float, required=False, default=0.0, help="Dropoutprobability. Default=0.0")
* ('--datadir', type=str, required=False, default="./data/", help="Datadirectory")
* ('--reg', type=float, default=1e-5, help="L2 regularization strength")
* ('--logdir', type=str, required=False, default="./logs/", help='Log directorypath')
* ('--modeldir', type=str, required=False, default="./models/", help='Modeldirectory path')
* ('--beta', type=float, default=0.5,help='The parameter for the dirichlet distribution for data partitioning')
* ('--device', type=str, default='cuda:0', help='The device to run the program')
* ('--log_file_name', type=str, default=None, help='The log file name')
* ('--optimizer', type=str, default='sgd', help='the optimizer')
* ('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
* ('--out_dim', type=int, default=256, help='the output dimension for theprojection layer')
* ('--temperature', type=float, default=0.5, help='the temperature parameter forcontrastive loss')
* ('--local_max_epoch', type=int, default=100, help='the number of epoch forlocal optimal training')
* ('--model_buffer_size', type=int, default=1, help='store how many previousmodels for contrastive loss')
* ('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
* ('--sample_fraction', type=float, default=1.0, help='how many clients aresampled in each round')
* ('--load_model_file', type=str, default=None, help='the model to load asglobal model')
* ('--load_pool_file', type=str, default=None, help='the old model pool path toload')
* ('--load_model_round', type=int, default=None, help='how many rounds haveexecuted for the loaded model')
* ('--load_first_net', type=int, default=1, help='whether load the first net asold net or not')
* ('--normal_model', type=int, default=0, help='use normal model or aggregatemodel')
* ('--loss', type=str, default='contrastive')
* ('--save_model',type=int,default=0)
* ('--use_project_head', type=int, default=1)
* ('--server_momentum', type=float, default=0, help='the server momentum(FedAvgM)')
# dataset 
if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:    n_classes = 10
elif args.dataset == 'celeba':    n_classes = 2
elif args.dataset == 'cifar100':    n_classes = 100
elif args.dataset == 'tinyimagenet':    n_classes = 200
elif args.dataset == 'femnist':    n_classes = 26
elif args.dataset == 'emnist':    n_classes = 47
elif args.dataset == 'xray': n_classes = 2