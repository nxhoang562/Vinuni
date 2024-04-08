# code to set up the model architecture, load the dataset, define the training loop, define optimizer
import sys
import argparse

import torch
import yaml
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from dataset import make_dataset, make_dataloader
from sklearn.model_selection import train_test_split
sys.path.append("/home/nxhoang/Work/HPE/src/denoise")

from model import AutoEncoder
from trainer import Trainer


def main(args):
    with open(args.config_path, 'r') as fd:  # change the .yaml file in your code.
        config = yaml.load(fd, Loader=yaml.FullLoader)

    # Create dataset
    train_dataset, test_dataset = make_dataset(config["data_root"], config)
    rng_generator = torch.manual_seed(config['init_rand_seed'])
    train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])
    val_data, test_data = train_test_split(test_dataset, test_size=0.5, random_state=41)
    val_loader = make_dataloader(val_data, is_training=False, generator=rng_generator, **config['validation_loader'])
    test_loader = make_dataloader(test_data, is_training=False, generator=rng_generator, **config['validation_loader'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    AE = AutoEncoder()
    AE = AE.to(device)
    AE.eval()

    optimizer = torch.optim.RMSprop(AE.parameters(), lr=config['lr'], momentum=config['momentum'])

    n_epochs = config['n_epochs']
    schedule = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 if epoch < 40 else torch.exp(-0.1))

    # Loss function
    criterion = nn.MSELoss().to(device)

    trainer = Trainer(AE, train_loader, val_loader, test_loader, criterion, optimizer, scheduler=schedule)

    for epoch in range(n_epochs):
        trainer.train_epoch()
        trainer.val_epoch()
    trainer.test_epoch()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config_path", type=str, required=True, help="Path to your config file")
#
#     args = parser.parse_args()
#     print(args)
#     main(args)
