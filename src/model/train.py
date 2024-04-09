import argparse
import sys
import torch
import yaml
import time



from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from model.ae_dataset import make_dataset, make_dataloader

from model.est_dataset import make_dataset as original_make_dataset
from model.est_dataset import make_dataloader as original_make_dataloader
from sklearn.model_selection import train_test_split

sys.path.append("/home/nxhoang/Work/HPE")
from src.denoise.model import AutoEncoder
from src.denoise.train import Trainer as Denoise_Trainer
from src.model.Denoise_Fdcnn import CombinedModel
from src.keypoint_detection.models.fdcnn import CNN
from src.keypoint_detection.trainer import Trainer as Estimastor_Trainer



def main(args):
    with open(args.config_path, 'r') as fd:  # change the .yaml file in your code.
        config = yaml.load(fd, Loader=yaml.FullLoader)

        # start = time.perf_counter()
        # Create noise dataset
        train_dataset, test_dataset = make_dataset(config["data_root"], config)
        # print(train_dataset[0][1].shape)
        rng_generator = torch.manual_seed(config['init_rand_seed'])
        train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])
        val_data, test_data = train_test_split(test_dataset, test_size=0.5, random_state=41)
        val_loader = make_dataloader(val_data, is_training=False, generator=rng_generator, **config['validation_loader'])
        test_loader = make_dataloader(test_data, is_training=False, generator=rng_generator, **config['validation_loader'])

        # Define device
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        # Initialize autoencoder
        autoencoder = AutoEncoder().to(device)
        # Train autoencoder
        criterion_ae = nn.MSELoss().to(device)
        optimizer_ae = torch.optim.RMSprop(autoencoder.parameters(), lr=config['lr'], momentum=config['momentum'])
        n_epochs_ae = config['n_epochs']
        schedule_ae = LambdaLR(optimizer_ae, lr_lambda=lambda epoch: 1 if epoch < 40 else torch.exp(-0.1))
        trainer_ae = Denoise_Trainer(autoencoder, train_loader, val_loader, test_loader, criterion=criterion_ae,
                                     optimizer=optimizer_ae, scheduler=schedule_ae)
        for epoch in range(n_epochs_ae):
            trainer_ae.train_epoch()
            trainer_ae.val_epoch()
        trainer_ae.test_epoch()

        # Extract Encoder part
        encoder = autoencoder.encoder
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False  # Freeze parameters

        # Initialize estimator
        estimator = CNN()

        # Initialize a combined model
        combined_model = CombinedModel(encoder, estimator).to(device)

        train_dataset1, test_dataset1 = original_make_dataset(config["data_root"], config)
        rng_generator1 = torch.manual_seed(config['init_rand_seed'])
        train_loader1 = original_make_dataloader(train_dataset1, is_training=True, generator=rng_generator1,
                                                 **config['train_loader'])
        val_data1, test_data1 = train_test_split(test_dataset1, test_size=0.5, random_state=41)
        val_loader1 = original_make_dataloader(val_data1, is_training=False, generator=rng_generator1,
                                               **config['validation_loader'])
        test_loader1 = original_make_dataloader(test_data1, is_training=False, generator=rng_generator1,
                                                **config['validation_loader'])

        # Training combined_model with Encoder parameters being frozen
        criterion_cb = nn.MSELoss().to(device)
        optimizer_cb = torch.optim.SGD(combined_model.parameters(), lr=config['lr_cb'], momentum=config['momentum_cb'])
        n_epochs_cb = config["n_epochs_cb"]
        n_epochs_decay = 60
        epoch_count = 1

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs_cb) / float(n_epochs_decay + 1)
            return lr_l

        scheduler_cb = torch.optim.lr_scheduler.LambdaLR(optimizer_cb, lr_lambda=lambda epoch: 1.0 - max(0,
                                                                                                         epoch + epoch_count - n_epochs_cb) / float(
            n_epochs_decay + 1))

        trainer_cb = Estimastor_Trainer(combined_model, train_loader1, val_loader1, test_loader1, criterion=criterion_cb,
                                        optimizer=optimizer_cb, scheduler=scheduler_cb)
        for epoch in range(n_epochs_cb):
            trainer_cb.train_epoch()
            trainer_cb.val_epoch()
        trainer_cb.test_epoch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to your config file")

    args = parser.parse_args()
    print(args)
    main(args)

