import os
import torch
from tqdm import tqdm
# import wandb
from metrics import compute_pck_pckh
import numpy as np


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer,
                 scheduler, model_save_path="checkpoints"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_save_path = model_save_path
        self.metric = dict()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_epoch_idx = 0
        self.val_epoch_idx = 0
        self.min_val_loss = 9999

    def train_epoch(self):
        print(f"******************************Training epoch {self.train_epoch_idx}************************************************")
        self.metric[self.train_epoch_idx] = dict()
        self.model.train()
        losses = []

        for iter, (data, gt) in enumerate(self.train_loader):
            # print(data.shape)
            data = np.transpose(data, (0, 1, 3, 2))
            # print('data_shape', data.shape)
            data = data.to(self.device)
            confidence = gt[:, :, 2:].to(self.device)
            # print('shape confidence', confidence.shape)
            label = gt[:, :, 0:2].to(self.device)
            # print('shape label', label.shape)
            predict = self.model(data)
            # print('shape predict', predict.shape)

            loss = self.criterion(torch.mul(predict, confidence), torch.mul(label, confidence)) / 32

            losses.append(loss.item())
            print(f"Iter {iter + self.train_epoch_idx * len(self.train_loader)}: MSE - {losses[-1]}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()

        self.metric[self.train_epoch_idx]["train_loss"] = sum(losses) / len(losses)
        self.train_epoch_idx += 1

    def val_epoch(self):
        print(f"*************************************Validating epoch {self.val_epoch_idx}************************************************")
        self.metric[self.train_epoch_idx] = dict()
        self.model.eval()

        pck_50_iter = []
        pck_40_iter = []
        pck_30_iter = []
        pck_20_iter = []
        pck_10_iter = []
        pck_5_iter = []

        losses = []
        for data, gt in self.val_loader:
            data = np.transpose(data, (0, 1, 3, 2)) # [batch size, channels, freq, time]
            data = data.to(self.device)
            label = gt[:, :, 0:2].to(self.device)
            confidence = gt[:, :, 2:].to(self.device)
            with torch.no_grad():
                predict = self.model(data)
            loss = self.criterion(torch.mul(confidence, predict), torch.mul(confidence, label))

            losses.append(loss.item())

            predict = predict.cpu()
            label = label.cpu()

            predict = torch.transpose(predict, 1, 2)
            label = torch.transpose(label, 1, 2)

            pck_50_iter.append(compute_pck_pckh(predict, label, 0.5))
            pck_40_iter.append(compute_pck_pckh(predict, label, 0.4))
            pck_30_iter.append(compute_pck_pckh(predict, label, 0.3))
            pck_20_iter.append(compute_pck_pckh(predict, label, 0.2))
            pck_10_iter.append(compute_pck_pckh(predict, label, 0.1))
            pck_5_iter.append(compute_pck_pckh(predict, label, 0.05))

        self.metric[self.val_epoch_idx]["val_loss"] = sum(losses) / len(losses)
        self.metric[self.val_epoch_idx]["pck_50"] = sum(pck_50_iter) / len(pck_50_iter)
        self.metric[self.val_epoch_idx]["pck_40"] = sum(pck_40_iter) / len(pck_40_iter)
        self.metric[self.val_epoch_idx]["pck_30"] = sum(pck_30_iter) / len(pck_30_iter)
        self.metric[self.val_epoch_idx]["pck_20"] = sum(pck_20_iter) / len(pck_20_iter)
        self.metric[self.val_epoch_idx]["pck_10"] = sum(pck_10_iter) / len(pck_10_iter)
        self.metric[self.val_epoch_idx]["pck_5"] = sum(pck_5_iter) / len(pck_5_iter)

        loss_avg = sum(losses) / len(losses)
        print(f"Val loss: {loss_avg}")
        print("pck_50: ", sum(pck_50_iter) / len(pck_50_iter))
        print("pck_40: ", sum(pck_40_iter) / len(pck_40_iter))
        print("pck_30: ", sum(pck_30_iter) / len(pck_30_iter))
        print("pck_20: ", sum(pck_20_iter) / len(pck_20_iter))
        print("pck_10: ", sum(pck_10_iter) / len(pck_10_iter))
        print("pck_5: ", sum(pck_5_iter) / len(pck_5_iter))

        # wandb.log({'Val loss': loss_avg})
        # wandb.log({'pck_50': sum(pck_50_iter) / len(pck_50_iter)})
        # wandb.log({'pck_40': sum(pck_40_iter) / len(pck_40_iter)})
        # wandb.log({'pck_30': sum(pck_30_iter) / len(pck_30_iter)})
        # wandb.log({'pck_20': sum(pck_20_iter) / len(pck_20_iter)})
        # wandb.log({'pck_10': sum(pck_10_iter) / len(pck_10_iter)})
        # wandb.log({'pck_5': sum(pck_5_iter) / len(pck_5_iter)})

        logs = "Val loss: " + str(sum(losses) / len(losses)) + ", "
        logs += "pck_50: " + str(sum(pck_50_iter) / len(pck_50_iter)) + ", "
        logs += "pck_40: " + str(sum(pck_40_iter) / len(pck_40_iter)) + ", "
        logs += "pck_30: " + str(sum(pck_30_iter) / len(pck_30_iter)) + ", "
        logs += "pck_20: " + str(sum(pck_20_iter) / len(pck_20_iter)) + ", "
        logs += "pck_10: " + str(sum(pck_10_iter) / len(pck_10_iter)) + ", "
        logs += "pck_5: " + str(sum(pck_5_iter) / len(pck_5_iter)) + "\n"

        with open("/home/nxhoang/Work/HPE/src/model/logs/combined_model.txt", "a") as f:

            f.write(logs)

        if self.val_epoch_idx == 0 or \
                self.metric[self.val_epoch_idx]["val_loss"] < self.min_val_loss:
            self.min_val_loss = self.metric[self.val_epoch_idx]["val_loss"]
            self.save_model()
            print(f"Model save at epoch {self.val_epoch_idx}")
        self.val_epoch_idx += 1

    def test_epoch(self):
        print(f"Testing")
        self.model.load_state_dict(torch.load("/home/nxhoang/Work/HPE/src/model/checkpoints/model_best.pth")['model_state_dict'])
        self.model.eval()
        pck_50_iter = []
        pck_40_iter = []
        pck_30_iter = []
        pck_20_iter = []
        pck_10_iter = []
        pck_5_iter = []
        losses = []
        for data, gt in tqdm(self.test_loader):
            data = np.transpose(data, (0, 1, 3, 2))  # [batch size, channels, freq, time]
            data = data.to(self.device)
            label = gt[:, :, 0:2].to(self.device)
            confidence = gt[:, :, 2:].to(self.device)
            with torch.no_grad():
                predict = self.model(data)
            loss = self.criterion(torch.mul(predict, confidence), torch.mul(label, confidence))

            losses.append(loss.item())

            predict = predict.cpu()
            label = label.cpu()
            predict = torch.transpose(predict, 1, 2)
            label = torch.transpose(label, 1, 2)

            pck_50_iter.append(compute_pck_pckh(predict, label, 0.5))
            pck_40_iter.append(compute_pck_pckh(predict, label, 0.4))
            pck_30_iter.append(compute_pck_pckh(predict, label, 0.3))
            pck_20_iter.append(compute_pck_pckh(predict, label, 0.2))
            pck_10_iter.append(compute_pck_pckh(predict, label, 0.1))
            pck_5_iter.append(compute_pck_pckh(predict, label, 0.05))

        self.metric["test"] = dict()
        self.metric["test"]["loss"] = sum(losses) / len(losses)
        self.metric["test"]["pck_50"] = sum(pck_50_iter) / len(pck_50_iter)
        self.metric["test"]["pck_40"] = sum(pck_40_iter) / len(pck_40_iter)
        self.metric["test"]["pck_30"] = sum(pck_30_iter) / len(pck_30_iter)
        self.metric["test"]["pck_20"] = sum(pck_20_iter) / len(pck_20_iter)
        self.metric["test"]["pck_10"] = sum(pck_10_iter) / len(pck_10_iter)
        self.metric["test"]["pck_5"] = sum(pck_5_iter) / len(pck_5_iter)

        print(f"Test result: {self.metric['test']}")
        print("pck_50: ", sum(pck_50_iter) / len(pck_50_iter))
        print("pck_40: ", sum(pck_40_iter) / len(pck_40_iter))
        print("pck_30: ", sum(pck_30_iter) / len(pck_30_iter))
        print("pck_20: ", sum(pck_20_iter) / len(pck_20_iter))
        print("pck_10: ", sum(pck_10_iter) / len(pck_10_iter))
        print("pck_5: ", sum(pck_5_iter) / len(pck_5_iter))

    def save_model(self):
        state_dict = dict()
        state_dict["model_state_dict"] = self.model.state_dict()
        state_dict["optimizer_state_dict"] = self.optimizer.state_dict()
        state_dict["train_loss_history"] = torch.tensor([self.metric[i]["train_loss"] for i in range(self.train_epoch_idx)])
        state_dict["val_loss_history"] = torch.tensor([self.metric[i]["val_loss"] for i in range(self.val_epoch_idx)])
        state_dict["pck_50"] = self.metric[self.val_epoch_idx]["pck_50"]
        state_dict["pck_40"] = self.metric[self.val_epoch_idx]["pck_40"]
        state_dict["pck_30"] = self.metric[self.val_epoch_idx]["pck_30"]
        state_dict["pck_20"] = self.metric[self.val_epoch_idx]["pck_20"]
        state_dict["pck_10"] = self.metric[self.val_epoch_idx]["pck_10"]
        state_dict["pck_5"] = self.metric[self.val_epoch_idx]["pck_5"]

        save_path = os.path.join(self.model_save_path, f"model_best.pth")
        torch.save(state_dict, save_path)
