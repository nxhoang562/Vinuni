
import os
import torch
from tqdm import tqdm


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

    def train_epoch(self):
        print(f"******************************Training epoch {self.train_epoch_idx}************************************************")
        self.metric[self.train_epoch_idx] = dict()
        self.model.train()
        losses = []
        for iter, (data, label) in enumerate(self.train_loader):
            data = data.to(self.device)
            label = label.to(self.device)
            predict = self.model(data)
            loss = self.criterion(predict, label)

            losses.append(loss.item())
            print(f"Iter {iter + self.train_epoch_idx * len(self.train_loader)}: MSE - {losses[-1]}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.metric[self.train_epoch_idx]["train_loss"] = sum(losses) / len(losses)
        self.train_epoch_idx += 1

    def val_epoch(self):
        print(f"*************************************Validating epoch {self.val_epoch_idx}************************************************")
        self.metric[self.train_epoch_idx] = dict()
        self.model.eval()
        losses = []
        for data, label in self.val_loader:
            data = data.to(self.device)
            label = label.to(self.device)
            with torch.no_grad():
                predict = self.model(data)
            loss = self.criterion(predict, label)

            losses.append(loss.item())

        self.metric[self.val_epoch_idx]["val_loss"] = sum(losses) / len(losses)
        print(f"Val loss: {sum(losses) / len(losses)}")
        if self.val_epoch_idx == 0 or \
                self.metric[self.val_epoch_idx]["val_loss"] < self.metric[self.val_epoch_idx - 1]["val_loss"]:
            self.save_model()
            print(f"Model save at epoch {self.val_epoch_idx}")
        self.val_epoch_idx += 1

    def test_epoch(self):
        print(f"Testing")
        self.model.eval()
        losses = []
        for data, label in tqdm(self.test_loader):
            data = data.to(self.device)
            label = label.to(self.device)
            with torch.no_grad():
                predict = self.model(data)
            loss = self.criterion(predict, label)

            losses.append(loss.item())

        self.metric["test"] = sum(losses) / len(losses)
        print(f"Test result: {self.metric['test']}")

    def save_model(self):
        state_dict = dict()
        state_dict["model_state_dict"] = self.model.state_dict()
        state_dict["optimizer_state_dict"] = self.optimizer.state_dict()
        state_dict["train_loss_history"] = torch.tensor([self.metric[i]["train_loss"] for i in range(self.train_epoch_idx)])
        state_dict["val_loss_history"] = torch.tensor([self.metric[i]["val_loss"] for i in range(self.val_epoch_idx)])

        save_path = os.path.join(self.model_save_path, f"model_best.pth")
        torch.save(state_dict, save_path)
