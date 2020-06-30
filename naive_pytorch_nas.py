"""
Example of using ray tune to find the optimal CNN architecture for mnist classification task
"""

import os, json, argparse, random, math, time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from filelock import FileLock

import ray
import ray.tune as tune
from ray.tune import Trainable
from ray.tune.utils import validate_save_restore
from ray.tune.schedulers import HyperBandScheduler, ASHAScheduler

IM_SIZE = 28
PADDING = 0
DILATION = 1

class Net(nn.Module):
    def __init__(self, net_config):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, net_config["c1_filter_num"], net_config["c1_filter_size"], net_config["c1_filter_stride"])
        conv1_output_size = math.floor((IM_SIZE + 2 * PADDING - DILATION * (net_config["c1_filter_size"] - 1) - 1) / net_config["c1_filter_stride"] + 1)
        conv1_out_channels = net_config["c1_filter_num"]

        self.conv2 = nn.Conv2d(net_config["c1_filter_num"], net_config["c2_filter_num"], net_config["c2_filter_size"], net_config["c2_filter_stride"])
        conv2_output_size = math.floor((conv1_output_size + 2 * PADDING - DILATION * (net_config["c2_filter_size"] - 1) - 1) / net_config["c2_filter_stride"] + 1)
        conv2_out_channels = net_config["c2_filter_num"]

        self.max_pool1 = nn.MaxPool2d(net_config["max_pool_kernel"])
        max_pool_output_size = math.floor((conv2_output_size + (2 * PADDING) - DILATION * (net_config["max_pool_kernel"] - 1) - 1) / net_config["max_pool_kernel"] + 1)
        max_pool_out_channels = conv2_out_channels

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(max_pool_out_channels * max_pool_output_size * max_pool_output_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(model, device, train_loader, optimizer):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return total_loss / (batch_idx+1)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    return test_loss

def get_data_loaders():
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/data.lock")):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "~/data",
                train=True,
                download=True,
                transform=mnist_transforms),
            batch_size=64,
            shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("~/data", train=False, transform=mnist_transforms),
        batch_size=64,
        shuffle=True)
    return train_loader, test_loader

class MyTrainableClass(Trainable):
    """Pytorch model trying to classify mnist handwritten digits
    """

    def _setup(self, config):
        use_cuda = config.get("use_gpu") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.batch_size = config["batch_size"]
        self.train_loader, self.test_loader = get_data_loaders()
        self.lr = config["lr"]
        self.model = Net(config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.timestep = 0

    def _train(self):
        self.timestep += 1
        train_loss = train(self.model, self.device, self.train_loader, self.optimizer)
        test_loss = test(self.model, self.device, self.test_loader)
        return {"train_loss": train_loss, "test_loss": test_loss}

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_args")
        with open(checkpoint_path, "w") as f:
            f.write(json.dumps({"timestep": self.timestep, "batch_size": self.batch_size, "lr": self.lr}))
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()}, os.path.join(checkpoint_dir, "checkpoint.pt"))
        return checkpoint_dir

    def _restore(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_args")
        with open(checkpoint_path) as f:
            foo = json.loads(f.read())
            self.timestep = foo["timestep"]
            self.batch_size = foo["batch_size"]
            self.lr = foo["lr"]
        # self.model = Net()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

parser = argparse.ArgumentParser()
parser.add_argument("--use-gpu", action="store_true", default=False, help="enables CUDA training")
parser.add_argument("--ray-address", type=str, help="The Redis address of the cluster.")
parser.add_argument("--smoke-test", action="store_true", help="Finish quickly for testing")
args = parser.parse_args()
ray.init(address=args.ray_address, num_cpus=6 if args.smoke_test else None)

# validate_save_restore(MyTrainableClass)
# validate_save_restore(MyTrainableClass, use_object_store=True)

hb = HyperBandScheduler(
    metric="test_loss",
    mode="min")

ahsa = ASHAScheduler(
    metric="test_loss")

analysis = tune.run(MyTrainableClass,
    name="pytorch_hparams_test",
    scheduler=ahsa,
    stop={"training_iteration": 3 if args.smoke_test else 10},
    num_samples=1 if args.smoke_test else 10,
    resources_per_trial={
        "cpu": 1,
        "gpu": int(args.use_gpu)
    },
    checkpoint_at_end=True,
    checkpoint_freq=3,
    config={
        "args": args,
        "batch_size": tune.sample_from(lambda _: int(np.random.random_integers(1, high=64))),
        "lr": tune.sample_from(lambda _: 10.0**np.random.random_integers(-2, high=2)),
        "c1_filter_num": tune.sample_from(lambda _: int(np.random.random_integers(1, high=64))),
        "c1_filter_size": tune.sample_from(lambda _: int(np.random.choice([3, 5]))),
        "c1_filter_stride": tune.sample_from(lambda _: int(np.random.choice([1, 2]))),
        "c2_filter_num": tune.sample_from(lambda _: int(np.random.random_integers(1, high=64))),
        "c2_filter_size": tune.sample_from(lambda _: int(np.random.choice([3, 5]))),
        "c2_filter_stride": tune.sample_from(lambda _: int(np.random.choice([1, 2]))),
        "max_pool_kernel": tune.sample_from(lambda _: int(np.random.choice([2])))
    },
    reuse_actors=True)

print("Best config is:", analysis.get_best_config(metric="test_loss"))
