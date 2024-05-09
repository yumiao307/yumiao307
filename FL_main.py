'''
Main file to set up the FL system and train
Code design inspired by https://github.com/FedML-AI/FedML
'''
import torch
#import wandb
import numpy as np
import random
import data_preprocessing.data_loader as dl
import argparse
from models.net import SimpleCNN, modVGG
from models.lenet5 import Lenet5
from torch.multiprocessing import set_start_method, Queue
import logging
import os
from collections import defaultdict
import time

# methods
import methods.fedavg as fedavg
import methods.fedprox as fedprox
import methods.scaffold as scaffold
import methods.fedgm as fedgm
import methods.fedmg as fedmg
import methods.moon as moon
import methods.fedmargin as fedmargin
import data_preprocessing.custom_multiprocess as cm
import methods.creff as creff
torch.multiprocessing.set_sharing_strategy('file_system')


def add_args(parser):
    # Training settings
    parser.add_argument('--method', type=str, default='creff', metavar='N',
                        help='Options are: fedavg, fedprox, scaffold')

    parser.add_argument('--data_dir', type=str, default='data/svhn',
                        help='data directory: data/fatigue, data/imgs, or another dataset')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local clients')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='alpha value for Dirichlet distribution partitioning of data(default: 0.5)')

    parser.add_argument('--partition_size', type=int, default=5000, metavar='PA',
                        help='alpha value for Dirichlet distribution partitioning of data(default: 0.5)')

    parser.add_argument('--client_number', type=int, default=10, metavar='NN',
                        help='number of clients in the FL system')

    parser.add_argument('--silos_number', type=int, default=1, metavar='NN',
                        help='number of silos in the FL system')

    parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--net', type=str, default='modVGG', metavar='N',
                        help='network arch')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--global_lr', type=float, default=1, metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.0001)

    parser.add_argument('--seed', help='random seed', type=int, default=1)

    parser.add_argument('--epochs', type=int, default=3, metavar='EP',
                        help='how many epochs will be trained locally per round')

    parser.add_argument('--match_epoch', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally per round')

    parser.add_argument('--comm_round', type=int, default=50,
                        help='how many rounds of communications are conducted')

    parser.add_argument('--mu', type=float, default=1, metavar='MU',
                        help='mu value for various methods')

    parser.add_argument('--lamb', type=float, default=0.1, metavar='MU',
                        help='mu value for various methods')
 
    parser.add_argument('--save_client', action='store_true', default=False,
                        help='Save client checkpoints each round')

    parser.add_argument('--thread_number', type=int, default=2, metavar='NN',
                        help='number of parallel training threads')

    parser.add_argument('--client_sample', type=float, default=0.2, metavar='MT',
                        help='Fraction of clients to sample')

    parser.add_argument('--gamma', type=float, default=0.5, metavar='MT',
                        help='Fraction of clients to sample')
                        
    parser.add_argument('--additional_experiment_name', default='', type=str,
                        help='hyperparameter gamma for mixup')
    parser.add_argument('--crt_epoch', type=int, default=300)
    parser.add_argument('--margin', type=float, default=1)
    args = parser.parse_args()

    return args


# Setup Functions
def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ## NOTE: If you want every run to be exactly the same each time
    ##       uncomment the following lines
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Helper Functions
def init_process(q, Client):
    set_random_seed()
    global client
    ci = q.get()
    client = Client(ci[0], ci[1])


def run_clients(received_info):
    try:
        return client.run(received_info)
    except KeyboardInterrupt:
        logging.info('exiting')
        return None


def allocate_clients_to_threads(args):
    mapping_dict = defaultdict(list)
    for round in range(args.comm_round):
        if args.client_sample < 1.0:
            num_clients = int(args.client_number * args.client_sample)
            client_list = random.sample(range(args.client_number), num_clients)
        else:
            num_clients = args.client_number
            client_list = list(range(num_clients))
        if num_clients % args.thread_number == 0 and num_clients > 0:
            clients_per_thread = int(num_clients / args.thread_number)
            for c, t in enumerate(range(0, num_clients, clients_per_thread)):
                idxs = [client_list[x] for x in range(t, t + clients_per_thread)]
                mapping_dict[c].append(idxs)
        else:
            raise ValueError("Sampled client number not divisible by number of threads")
    return mapping_dict


if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    # get arguments
    #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    set_random_seed(args.seed)
    # wandb initialization
    # experiment_name = 'seed:' + str(args.seed)
    # print(experiment_name)
    # wandb_log_dir = os.path.join('./fed/wandb', experiment_name)
    # if not os.path.exists('{}'.format(wandb_log_dir)):
    #     os.makedirs('{}'.format(wandb_log_dir))
    # group = '{}_{}_{}_{}_epochs:{}_client:{}_fraction:{}'.format(args.data_dir.split('/')[-1], args.partition_method, args.net, (
    #               str(args.partition_alpha) if args.partition_method == 'hetero' else ""), str(args.epochs),
    #                                      str(args.client_number), str(args.client_sample))
    # group = '{}_{}_{}'.format(args.data_dir.split('/')[-1], args.partition_method, args.net)
    # wandb.init(entity='lxjxlxj', project='FL',
    #            group=group, settings=wandb.Settings(disable_git=True, save_code=False),
    #            job_type=args.method + (
    #                "_" + args.additional_experiment_name if args.additional_experiment_name != '' else ''),
    #            dir=wandb_log_dir)
    #
    # wandb.run.name = experiment_name + ', mu:{}'.format(args.mu)
    # wandb.run.save()
    # wandb.config.update(args)
    # get data
    _, _, train_data_global, test_data_global, _, train_data_local_dict, test_data_local_dict, class_num = \
        dl.load_partition_data(args.data_dir, args.partition_method, args.partition_alpha, args.client_number, args.batch_size, args.partition_size)
    # _, _, train_data_global, test_data_global, _, train_data_local_dict, test_data_local_dict, class_num = \
    #     dl.load_partition_data2(args)
    mapping_dict = allocate_clients_to_threads(args)
    # init method and model type
    print(args.method)
    if args.method == 'fedavg':
        Server = fedavg.Server
        Client = fedavg.Client
        Model = eval(args.net)
        server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': Model,
                       'num_classes': class_num}
        client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict,
                        'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in
                       range(args.thread_number)]
    elif args.method == 'fedprox':
        Server = fedprox.Server
        Client = fedprox.Client
        Model = eval(args.net)
        server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': Model,
                       'num_classes': class_num}
        client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict,
                        'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in
                       range(args.thread_number)]
    elif args.method == 'fedgm':
        Server = fedgm.Server
        Client = fedgm.Client
        Model = eval(args.net)
        server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': Model,
                       'num_classes': class_num}
        client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict,
                        'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in
                       range(args.thread_number)]
    elif args.method == 'fedmg':
        Server = fedmg.Server
        Client = fedmg.Client
        Model = eval(args.net)
        server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': Model,
                       'num_classes': class_num}
        client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict,
                        'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in
                       range(args.thread_number)]
    elif args.method == 'scaffold':
        Server = scaffold.Server
        Client = scaffold.Client
        Model = eval(args.net)
        server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': Model,
                       'num_classes': class_num}
        client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict,
                        'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in
                       range(args.thread_number)]
    elif args.method == 'moon':
        Server = moon.Server
        Client = moon.Client
        Model = eval(args.net)
        server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': Model,
                       'num_classes': class_num}
        client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict,
                        'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in
                       range(args.thread_number)]
    elif args.method == 'fedmargin':
        Server = fedmargin.Server
        Client = fedmargin.Client
        Model = eval(args.net)
        server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': Model,
                       'num_classes': class_num}
        client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict,
                        'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in
                       range(args.thread_number)]
    elif args.method == 'creff':
        Server = creff.Server
        Client = creff.Client
        Model = eval(args.net)
        server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': Model,
                       'num_classes': class_num}
        client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict,
                        'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in
                       range(args.thread_number)]
    else:
        raise ValueError('Invalid --method chosen! Please choose from availible methods.')

    # init nodes
    client_info = Queue()
    for i in range(args.thread_number):
        client_info.put((client_dict[i], args, i+1))

    # Start server and get initial outputs
    pool = cm.DreamPool(args.thread_number, init_process, (client_info, Client))
    # init server
    server_dict['save_path'] = '{}/logs/{}__{}_e{}_c{}'.format(os.getcwd(),
                                                               time.strftime("%Y%m%d_%H%M%S"), args.method, args.epochs,
                                                               args.client_number)
    if not os.path.exists(server_dict['save_path']):
        os.makedirs(server_dict['save_path'])
    server = Server(server_dict, args)
    server_outputs = server.start()
    # Start Federated Training
    time.sleep(60 * (args.thread_number / 2))  # Allow time for threads to start up
    for r in range(args.comm_round):
        logging.info('************** Round: {} ***************'.format(r))
        round_start = time.time()
        client_outputs = pool.map(run_clients, server_outputs)
        client_outputs = [c for sublist in client_outputs for c in sublist]
        server_outputs = server.run(client_outputs)
        round_end = time.time()
        logging.info('Round {} Time: {}s'.format(r, round_end - round_start))
    pool.close()
    pool.join()

  #  wandb.finish()
