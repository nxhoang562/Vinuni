import argparse

import torch
import yaml
from torch import nn
import parser

from dataset import make_dataset, make_dataloader
from sklearn.model_selection import train_test_split
from models.detransformer import DeTransformer
from src.denoise.model import AutoEncoder
# from models.baseline import PoseNet
# from models.baseline_dcn import PoseNet

from models.fdcnn import CNN
from trainer import Trainer
from models.informer import Informer


def main(args):
    with open(args.config_path, 'r') as fd:  # change the .yaml file in your code.
        config = yaml.load(fd, Loader=yaml.FullLoader)

    # Create dataset
    train_dataset, test_dataset = make_dataset(config["data_root"], config)
    # print(train_dataset[0][0].shape)
    rng_generator = torch.manual_seed(config['init_rand_seed'])
    train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])
    val_data, test_data = train_test_split(test_dataset, test_size=0.5, random_state=41)
    val_loader = make_dataloader(val_data, is_training=False, generator=rng_generator, **config['validation_loader'])
    test_loader = make_dataloader(test_data, is_training=False, generator=rng_generator, **config['validation_loader'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    # denoiser = AutoEncoder()
    # denoiser.load_state_dict(torch.load("/home/nxhoang/HPE/src/denoise/checkpoints/model_best.pth")["model_state_dict"])
    # detector = DeTransformer(denoiser, config["d_input"], config['d_model'], config['d_output'], config['q'],
    #                          config['v'], config['h'], config['N'], config['attention_size'],
    #                          config['dropout'], config['chunk_mode'], config['pe'], config['pe_period'])
    # detector = detector.to(device)
    detector = CNN().to(device)
    # detector.load_state_dict(torch.load("/home/nxhoang/HPE/src/keypoint_detection/checkpoints/model_best.pth")["model_state_dict"])
    # detector = torch.load("/home/nxhoang/HPE/metafi/get_train.py").to(device)
    # detector = Informer(enc_in=136, c_out=17).to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(detector.parameters(), lr=config['lr'], momentum=config['momentum'])
    n_epochs = config["n_epochs"]
    n_epochs_decay = 60
    epoch_count = 1

    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
        return lr_l

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 - max(0,
                                                                                               epoch + epoch_count - n_epochs) / float(
        n_epochs_decay + 1))

    trainer = Trainer(detector, train_loader, val_loader, test_loader, criterion, optimizer, scheduler=scheduler)
    for epoch in range(n_epochs):
        trainer.train_epoch()
        trainer.val_epoch()
    trainer.test_epoch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to your config file")

    args = parser.parse_args()
    print(args)
    main(args)
