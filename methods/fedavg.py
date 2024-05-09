import torch
import wandb
from methods.base import Base_Client, Base_Server


class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes, KD=True).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.wd)


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes, KD=True)
        # wandb.watch(self.model)
