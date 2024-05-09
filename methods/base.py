import torch
import logging
import json
import wandb
from torch.multiprocessing import current_process


class Base_Client():
    def __init__(self, client_dict, args):
        self.train_data = client_dict['train_data']
        self.test_data = client_dict['test_data']
        self.device = 'cuda:{}'.format(client_dict['device'])
        self.model_type = client_dict['model_type']
        self.num_classes = client_dict['num_classes']
        self.args = args
        self.round = 0
        self.client_map = client_dict['client_map']
        self.train_dataloader = None
        self.test_dataloader = None
        self.client_index = None

    def load_client_state_dict(self, server_state_dict):
        # If you want to customize how to state dict is loaded you can do so here
        self.model.load_state_dict(server_state_dict)

    def init_client_infos(self):
        client_cnts = torch.zeros(self.num_classes).float()
        for _, labels in self.train_dataloader:
            for label in labels.numpy():
                client_cnts[label] += 1
        return client_cnts

    def get_dist(self):
        self.client_cnts = self.init_client_infos()
        dist = self.client_cnts / self.client_cnts.sum()  # 个数的比例

        return dist

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
        self.model.train()
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                h, log_probs = self.model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info(
                    '(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                                                    epoch, sum(
                            epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        weights = self.model.cpu().state_dict()
        return weights

    def test(self):
        self.model.to(self.device)
        self.model.eval()

        wandb_dict = {}
        test_correct = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.test_dataloader):
                x = x.to(self.device)
                target = target.to(self.device)

                pred = self.model(x)
                if type(pred) == tuple:
                    hs, pred = pred
                # loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                # test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)
            acc = (test_correct / test_sample_number) * 100
            wandb_dict[self.args.method + "_clinet:{}_acc".format(self.client_index)] = acc
            # wandb_dict[self.args.method + "_loss"] = loss
            # wandb.log(wandb_dict)
            logging.info(
                "************* Round {} Client {} Acc = {:.2f} **************".format(self.round, self.client_index,
                                                                                      acc))

        return acc

class Base_Server():
    def __init__(self, server_dict, args):
        self.train_data = server_dict['train_data']
        self.test_data = server_dict['test_data']
        self.device = 'cuda:{}'.format(torch.cuda.device_count() - 1)
        self.model_type = server_dict['model_type']
        self.num_classes = server_dict['num_classes']
        self.acc = 0.0
        self.round = 0
        self.args = args
        self.criterion = torch.nn.CrossEntropyLoss()
        self.save_path = server_dict['save_path']

    def run(self, received_info):
        server_outputs = self.operations(received_info)
        acc = self.test()
        self.log_info(received_info, acc)
        #self.round+=1
        if acc > self.acc:
            torch.save(self.model.state_dict(), '{}/{}.pt'.format(self.save_path, 'server'))
            self.acc = acc
        return server_outputs

    def start(self):
        with open('{}/config.txt'.format(self.save_path), 'a+') as config:
            config.write(json.dumps(vars(self.args)))
        return [self.model.cpu().state_dict() for _ in range(self.args.thread_number)]

    def log_info(self, client_info, acc):
        client_acc = sum([c['acc'] for c in client_info]) / len(client_info)
        out_str = 'Test/AccTop1: {}, Client_Train/AccTop1: {}, round: {}\n'.format(acc, client_acc, self.round)
        with open('{}/out.log'.format(self.save_path), 'a+') as out_file:
            out_file.write(out_str)

    def _flatten_weights_from_model(self, model, device):
        """Return the weights of the given model as a 1-D tensor"""
        weights = torch.tensor([], requires_grad=False).to(device)
        model.to(device)
        for param in model.parameters():
            weights = torch.cat((weights, torch.flatten(param)))
        return weights

    def operations(self, client_info):

        client_info.sort(key=lambda tup: tup['client_index'])
        client_sd = [c['weights'] for c in client_info]
        cw = [c['num_samples'] / sum([x['num_samples'] for x in client_info]) for c in client_info]

        ssd = self.model.state_dict()
        for key in ssd:
            ssd[key] = sum([sd[key] * cw[i] for i, sd in enumerate(client_sd)])
        self.model.load_state_dict(ssd)
        if self.args.save_client:
            for client in client_info:
                torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        self.round += 1
        return [self.model.cpu().state_dict() for _ in range(self.args.thread_number)]

    def test(self):
        self.model.to(self.device)
        self.model.eval()
        hs = None
        labelss = None
        preds = None

        wandb_dict = {}
        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.test_data):
                x = x.to(self.device)
                target = target.to(self.device)

                pred = self.model(x)
                if type(pred) == tuple:
                    h, pred = pred
                loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)
                hs = h.detach() if hs is None else torch.cat([hs, h.detach().clone()], dim=0)
                labelss = target if labelss is None else torch.cat([labelss, target.clone()], dim=0)
                preds = predicted.detach() if preds is None else torch.cat([preds, predicted.detach().clone()], dim=0)

            acc = (test_correct / test_sample_number) * 100
            loss = (test_loss / test_sample_number)
            # wandb_dict[self.args.method + "_acc".format(self.args.mu)] = acc
            # wandb_dict[self.args.method + "_loss".format(self.args.mu)] = loss
            # wandb.log(wandb_dict)

            logging.info("************* Server Acc = {:.2f} **************".format(acc))
        return acc