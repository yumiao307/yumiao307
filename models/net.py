import torch
import torch.nn as nn
import torch.nn.functional as F

from models.MarginLinear import LSoftmaxLinear, SoftmaxMargin, DisAlignLinear, LGMLoss_v0, SoftmaxMarginMix

class modVGG(nn.Module):
    def __init__(self, n_classes: int = 10, KD=False, projection=False, margin=0):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            nn.Flatten(),
            nn.Linear(4 * 4 * 256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.clf = nn.Linear(512, n_classes) if not margin else SoftmaxMargin(512, n_classes, margin=margin)
        self.KD = KD
        self.projection = projection
        self.margin = margin != 0
        self.relu = nn.ReLU()

        if projection:
            self.p1 = nn.Linear(512 * 1, 512 * 1)
            self.p2 = nn.Linear(512 * 1, 256)
            self.clf = nn.Linear(256, n_classes)

    def forward(self, X: torch.Tensor, target=None):
        x_f = self.seq(X)
        if self.projection:
            x_p = self.p1(x_f)
            x_p = self.relu(x_p)
            x_f = self.p2(x_p)
        X = self.clf(x_f) if not self.margin else self.clf(x_f, target)

        if self.KD == True:
            return x_f, X
        else:
            return X

    def vis(self, X):
        X = self.seq(X)
        return X


class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int = 10, KD=False, projection=False, margin=0):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)  # 28x28
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 10x10
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 5x5
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.clf = nn.Linear(84, n_classes) if not margin else SoftmaxMargin(84, n_classes, margin=margin)
        self.relu = nn.ReLU()

        self.KD = KD
        self.projection = projection
        self.margin = margin != 0

        if projection:
            self.p1 = nn.Linear(84 * 1, 84 * 1)
            self.p2 = nn.Linear(84 * 1, 256)
            self.clf = nn.Linear(256, n_classes)

    def forward(self, X: torch.Tensor, target=None):
        X = self.pool1(F.relu(self.conv1(X)))
        X = self.pool2(F.relu(self.conv2(X)))
        X = self.flatten(X)
        X = F.relu(self.fc1(X))
        x_f = F.relu(self.fc2(X))

        if self.projection:
            x_p = self.p1(x_f)
            x_p = self.relu(x_p)
            x_f = self.p2(x_p)
        X = self.clf(x_f) if not self.margin else self.clf(x_f, target)

        if self.KD == True:
            return x_f, X
        else:
            return X

    def vis(self, X):
        X = self.pool1(F.relu(self.conv1(X)))
        X = self.pool2(F.relu(self.conv2(X)))
        X = self.flatten(X)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        return X


if __name__ == '__main__':
    # model = SimpleCNN()
    model = modVGG()
    print(sum(param.numel() for param in model.parameters()))
