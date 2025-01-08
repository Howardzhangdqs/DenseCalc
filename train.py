from utils_ml import train, validate
# from utils_ml import AutoOptimizerSwitcher
from utils_ml import DenseCalcDataset
from torch.utils.data import DataLoader
from loss import Loss
from model import Model

import transformers
from lion_pytorch import Lion


N = 5
EPOCHS = 100
WARMUP_EPOCHS = 1
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-2
OPTIMIZER_SWITCHER_PATIENCE = 10


def main():

    DenseCalcDataset(N=5)

    model = Model()
    criterion = Loss()
    # optimizer = AutoOptimizerSwitcher(
    #     model,
    #     lr=LR,
    #     weight_decay=WEIGHT_DECAY,
    #     patience=OPTIMIZER_SWITCHER_PATIENCE
    # )

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

    for epoch in range(EPOCHS):
        train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            epoch=(epoch, EPOCHS)
        )

        validate(
            model=model,
            val_loader=test_loader,
            criterion=criterion,
            epoch=(epoch, EPOCHS)
        )


if __name__ == "__main__":
    main()
