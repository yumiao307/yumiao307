import torch
import torch.nn as nn
import json
import wandb
import logging
from methods.base import Base_Client, Base_Server
import copy
from torch.multiprocessing import current_process

__all__ = ["flatten_weights", "flatten_grads", "assign_weights", "assign_grads"]


def flatten_weights(model, numpy_output=True):
    """
    Flattens a PyTorch model. i.e., concat all parameters as a single, large vector.
    :param model: PyTorch model
    :param numpy_output: should the output vector be casted as numpy array?
    :return: the flattened (vectorized) model parameters either as Numpy array or Torch tensors
    """
    all_params = []
    for param in model.parameters():
        all_params.append(param.view(-1))
    all_params = torch.cat(all_params)
    if numpy_output:
        return all_params.cpu().detach().numpy()
    return all_params


def flatten_grads(model):
    """
    Flattens the gradients of a model (after `.backward()` call) as a single, large vector.
    :param model: PyTorch model.
    :return: 1D torch Tensor
    """
    all_grads = []
    for name, param in model.named_parameters():
        all_grads.append(param.grad.view(-1))
    return torch.cat(all_grads)

def flatten_variate(model):
    """
    Flattens the gradients of a model (after `.backward()` call) as a single, large vector.
    :param model: PyTorch model.
    :return: 1D torch Tensor
    """
    all_grads = []
    for name, value in model.items():
        all_grads.append(value.view(-1))
    return torch.cat(all_grads)

def assign_weights(model, weights):
    """
    Manually assigns `weights` of a Pytorch `model`.
    Note that weights is of vector form (i.e., 1D array or tensor).
    Usage: For implementation of Mode Connectivity SGD algorithm.
    :param model: Pytorch model.
    :param weights: A flattened (i.e., 1D) weight vector.
    :return: The `model` updated with `weights`.
    """
    state_dict = model.state_dict(keep_vars=True)
    # The index keeps track of location of current weights that is being un-flattened.
    index = 0
    # just for safety, no grads should be transferred.
    with torch.no_grad():
        for param in state_dict.keys():
            # ignore batchnorm params
            if (
                "running_mean" in param
                or "running_var" in param
                or "num_batches_tracked" in param
            ):
                continue
            param_count = state_dict[param].numel()
            param_shape = state_dict[param].shape
            state_dict[param] = nn.Parameter(
                torch.from_numpy(
                    weights[index : index + param_count].reshape(param_shape)
                )
            )
            index += param_count
    model.load_state_dict(state_dict)
    return model


def assign_grads(model, grads):
    """
    Similar to `assign_weights` but this time, manually assign `grads` vector to a model.
    :param model: PyTorch Model.
    :param grads: Gradient vectors.
    :return:
    """
    state_dict = model.state_dict(keep_vars=True)
    index = 0
    for param in state_dict.keys():
        # ignore batchnorm params
        if (
            "running_mean" in param
            or "running_var" in param
            or "num_batches_tracked" in param
        ):
            continue
        param_count = state_dict[param].numel()
        param_shape = state_dict[param].shape
        state_dict[param].grad = (
            grads[index : index + param_count].view(param_shape).clone()
        )
        index += param_count
    model.load_state_dict(state_dict)
    return model

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.step_count = 0
        self.c, self.c_i = 0, 0
        self.model = self.model_type(self.num_classes, KD=False).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)


    def load_client_state_dict(self, server_state_dict, client_index):
        # If you want to customize how to state dict is loaded you can do so here
        self.model.load_state_dict(server_state_dict[0])
        self.c = copy.deepcopy(server_state_dict[1])
        self.c_i = server_state_dict[2][client_index]

    def run(self, received_info):
        client_results = []
        for client_idx in self.client_map[self.round]:
            self.load_client_state_dict(received_info, client_idx)
            self.train_dataloader = self.train_data[client_idx]
            self.test_dataloader = self.test_data[client_idx]
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None and self.train_dataloader._iterator._shutdown:
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()
            self.client_index = client_idx
            num_samples = len(self.train_dataloader) * self.args.batch_size
            weights, new_c = self.train()
            c = copy.deepcopy(new_c)
            for name in new_c.keys():
                c[name] = new_c[name] - self.c_i[name]
                self.c_i[name] = new_c[name]
            acc = self.test()
            client_results.append(
                {'weights': weights, 'num_samples': num_samples, 'acc': acc, 'client_index': self.client_index,\
                 'c_update': c, 'c': self.c_i})
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()

        self.round += 1
        # self.tsne_vis()
        return client_results

    def train(self):
        # train the local model
        self.model.to(self.device)
        self.model.train()
        sd = copy.deepcopy(self.model.state_dict())
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                loss = self._scaffold_step(images, labels)
                batch_loss.append(loss)
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info(
                    '(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                                                    epoch, sum(
                            epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))

        # update control variates for scaffold algorithm
        ssd = self.model.state_dict()
        new_c = copy.deepcopy(self.c_i)
        for name in sd.keys():
            new_c[name] = self.c_i[name] - self.c[name] + \
                                 (sd[name].cpu() - ssd[name].cpu()) / (self.args.epochs * len(self.train_dataloader) * self.args.lr)

        weights = self.model.cpu().state_dict()
        return weights, new_c

    def _scaffold_step(self, data, targets):
        self.optimizer.zero_grad()

        # forward pass
        data, targets = data.to(self.device), targets.to(self.device)
        logits = self.model(data)
        loss = self.criterion(logits, targets)

        # backward pass
        loss.backward()
        grad_batch = flatten_grads(self.model).detach().clone()
        self.optimizer.zero_grad()

        # add control variate
        grad_batch = grad_batch - flatten_variate(self.c_i).to(self.device) + flatten_variate(self.c).to(self.device)
        self.model = assign_grads(self.model, grad_batch)
        self.optimizer.step()
        self.step_count += 1
        return loss.item()


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super(Server, self).__init__(server_dict, args)
        self.model = self.model_type(self.num_classes, KD=True)
        wandb.watch(self.model)
        self.c, self.local_c = self._init_control_variates()

    def _init_control_variates(self):
        c = self.model.state_dict()
        for name in c.keys():
            c[name] = torch.zeros(c[name].shape)

        local_c = []

        for _ in range(self.args.client_number):
            local_c.append(copy.deepcopy(c))

        return c, local_c

    def operations(self, client_info):

        client_info.sort(key=lambda tup: tup['client_index'])
        client_sd = [c['weights'] for c in client_info]
        cw = [c['num_samples'] / sum([x['num_samples'] for x in client_info]) for c in client_info]
        cc_update = [c['c_update'] for c in client_info]
        cc = [c['c'] for c in client_info]
        client_indices = [c['client_index'] for c in client_info]

        ssd = self.model.cpu().state_dict()
        copy_ssd = copy.deepcopy(self.model.cpu().state_dict())
        for key in ssd:
            ssd[key] += sum([(sd[key] - copy_ssd[key]) * cw[i] * self.args.global_lr for i, sd in enumerate(client_sd)])
        self.model.load_state_dict(ssd)
        if self.args.save_client:
            for client in client_info:
                torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        self.round += 1
        # global c update
        c = copy.deepcopy(cc[0])
        for key in self.c.keys():
            c[key] = sum([update[key] * cw[i] for i, update in enumerate(cc_update)])

        # global c
        for key in self.c.keys():
            self.c[key] += c[key] * self.args.client_sample
        # local c
        for client_idx, c in zip(client_indices, cc):
            self.local_c[client_idx] = c

        return [(self.model.cpu().state_dict(), self.c, self.local_c) for _ in range(self.args.thread_number)]

    def run(self, received_info):
        server_outputs = self.operations(received_info)
        acc = self.test()
        self.log_info(received_info, acc)
        if acc > self.acc:
            torch.save(self.model.state_dict(), '{}/{}.pt'.format(self.save_path, 'server'))
            self.acc = acc
        return server_outputs

    def start(self):
        with open('{}/config.txt'.format(self.save_path), 'a+') as config:
            config.write(json.dumps(vars(self.args)))
        return [(self.model.cpu().state_dict(), self.c, self.local_c) for _ in range(self.args.thread_number)]