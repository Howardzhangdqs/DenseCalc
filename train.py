from utils_ml import train, validate
from utils_ml import AutoOptimizerSwitcher
from utils_ml import DenseCalcDataset
from torch.utils.data import DataLoader
from loss import Loss
from model import Model


N = 5
EPOCHS = 100
WARMUP_EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-3
WEIGHT_DECAY = 1e-2
OPTIMIZER_SWITCHER_PATIENCE = 10


def main():

    DenseCalcDataset(N=5)

    model = Model()
    criterion = Loss()
    optimizer = AutoOptimizerSwitcher(
        model,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        patience=OPTIMIZER_SWITCHER_PATIENCE
    )

    train_dataset = DenseCalcDataset(train=True, N=N)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=8)

    test_dataset = DenseCalcDataset(train=False, N=N)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=8)

    scheduler = optimizer.get_scheduler(total_steps=EPOCHS, warmup_steps=WARMUP_EPOCHS)

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
