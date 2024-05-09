
import torch
import json
import wandb
import logging
from methods.base import Base_Client, Base_Server
import copy
from collections import OrderedDict
from torch.multiprocessing import current_process

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes, KD=True).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        self.delta = None


    def load_client_state_dict(self, server_state_dict):
        # If you want to customize how to state dict is loaded you can do so here
        self.model.load_state_dict(server_state_dict[0])
        self.delta = server_state_dict[1]
    
    def run(self, received_info):
        client_results = []
        for client_idx in self.client_map[self.round]:
            self.load_client_state_dict(received_info)
            self.train_dataloader = self.train_data[client_idx]
            self.test_dataloader = self.test_data[client_idx]
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None and self.train_dataloader._iterator._shutdown:
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()
            self.client_index = client_idx
            num_samples = len(self.train_dataloader) * self.args.batch_size
            weights = self.train()
            acc = self.test()
            client_results.append(
                {'weights': weights, 'num_samples': num_samples, 'acc': acc, 'client_index': self.client_index})
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()

        self.round += 1
        # self.tsne_vis()
        return client_results

    def train(self):
        # train the local model
        self.model.to(self.device)
        global_weight_collector = copy.deepcopy(list(self.model.parameters()))
        self.model.train()
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                
                sd = self.model.state_dict()
                for name, value in sd.items():
                    sd[name] += self.args.mu * self.delta[name].to(self.device)
                self.model.load_state_dict(sd)

                h, log_probs = self.model(images)
                loss = self.criterion(log_probs, labels)

                ########
                loss.backward()
                if self.round > 0:
                    for param, name in zip(self.model.parameters(), self.delta.keys()):
                        param.grad.data.mul_(1)
                self.optimizer.step()
                batch_loss.append(loss.item())

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                            epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        weights = self.model.cpu().state_dict()
        return weights

class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes, KD=True)
        # wandb.watch(self.model)

    def operations(self, client_info):

        client_info.sort(key=lambda tup: tup['client_index'])
        client_sd = [c['weights'] for c in client_info]
        cw = [c['num_samples'] / sum([x['num_samples'] for x in client_info]) for c in client_info]

        delta = self.model.cpu().state_dict()
        ssd = self.model.state_dict()
        copy_ssd = copy.deepcopy(self.model.cpu().state_dict())
        
        for key in ssd:
            ssd[key] += sum([(sd[key] - copy_ssd[key]) * cw[i] * self.args.global_lr for i, sd in enumerate(client_sd)])

        self.model.load_state_dict(ssd)
        
        for name, value in delta.items():
            delta[name] = (ssd[name] - copy_ssd[name]) / self.args.epochs / (self.args.partition_size/ self.args.batch_size)
        if self.args.save_client:
            for client in client_info:
                torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        self.round += 1

        return [(self.model.cpu().state_dict(), delta) for _ in range(self.args.thread_number)]

    def start(self):
        with open('{}/config.txt'.format(self.save_path), 'a+') as config:
            config.write(json.dumps(vars(self.args)))
        delta = OrderedDict()
        for name, value in self.model.state_dict().items():
            delta[name] = torch.zeros(value.shape)
        return [(self.model.cpu().state_dict(), delta) for _ in range(self.args.thread_number)]