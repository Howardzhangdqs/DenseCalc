import torch
import os
from tqdm import tqdm
import wandb

from utils_ml import train, validate
# from utils_ml import AutoOptimizerSwitcher
from utils_ml import DenseCalcDataset
from torch.utils.data import DataLoader
from loss import Loss
from model import Model

import transformers
from lion_pytorch import Lion


N = 5
EPOCHS = 50
WARMUP_EPOCHS = int(EPOCHS / 10)
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-2
OPTIMIZER_SWITCHER_PATIENCE = 10
OPTIMIZER = "adamw"


wandb_instance = wandb.init(project="my-project")
wandb.config.update({
    "N": N,
    "epochs": EPOCHS,
    "warmup_epochs": WARMUP_EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "weight_decay": WEIGHT_DECAY,
    "optimizer_switcher_patience": OPTIMIZER_SWITCHER_PATIENCE,
})


def main():

    DenseCalcDataset(N=5)

    model = Model()
    # optimizer = AutoOptimizerSwitcher(
    #     model,
    #     lr=LR,
    #     weight_decay=WEIGHT_DECAY,
    #     patience=OPTIMIZER_SWITCHER_PATIENCE
    # )

    if OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )
    elif OPTIMIZER == "lion":
        optimizer = Lion(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )

    train_dataset = DenseCalcDataset(train=True, N=N)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=8)

    test_dataset = DenseCalcDataset(train=False, N=N)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=8)

    # scheduler = optimizer.get_scheduler(total_steps=EPOCHS * len(train_loader), warmup_steps=0)
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_EPOCHS * len(train_loader),
        num_training_steps=EPOCHS * len(train_loader)
    )

    # distribution of train data
    label_num = [0, 0]
    for _, targets in tqdm(train_loader, desc="Checking data distribution"):
        targets = torch.stack(targets, dim=1).float()
        targets = targets[:, :2].float()
        label_num[0] += targets[:, 0].sum().item()
        label_num[1] += targets[:, 1].sum().item()

    label_num = [label_num[1] / sum(label_num), label_num[0] / sum(label_num)]

    print(f"Train data distribution: {label_num}")

    criterion = Loss(distribution=label_num)

    last_acc = 0

    for epoch in range(EPOCHS):
        train_loss = train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            epoch=(epoch, EPOCHS)
        )

        test_loss, acc = validate(
            model=model,
            val_loader=test_loader,
            criterion=criterion,
            epoch=(epoch, EPOCHS)
        )

        if acc > last_acc:
            last_acc = acc
            os.makedirs("runs", exist_ok=True)
            torch.save(model.state_dict(), f"runs/model_{wandb_instance.name}_{epoch:03}_{acc*1000:.0f}.pth")

        wandb.log({
            "train_loss": train_loss,
            "test_loss": test_loss,
            "acc": acc
        })


if __name__ == "__main__":
    main()
