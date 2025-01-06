from utils_ml import train, validate
from utils_ml import AutoOptimizerSwitcher
from transformers import get_cosine_schedule_with_warmup
from model import Model

from utils_ml import DenseCalcDataset
from torch.utils.data import DataLoader
from loss import Loss


N = 5
EPOCHS = 100
BATCH_SIZE = 128
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
    scheduler = optimizer.get_scheduler(total_steps=1000, warmup_steps=100)

    train_dataset = DenseCalcDataset(train=True, N=N)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    test_dataset = DenseCalcDataset(train=False, N=N)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

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
