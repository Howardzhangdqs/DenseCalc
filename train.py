from utils_ml import train, validate
from utils_ml import AutoOptimizerSwitcher
from transformers import get_cosine_schedule_with_warmup
from model import model

from utils_ml import DenseCalcDataset


N = 5
EPOCHS = 100
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-2
OPTIMIZER_SWITCHER_PATIENCE = 10


def main():

    DenseCalcDataset(N=5)

    model = model()
    optimizer = AutoOptimizerSwitcher(
        model,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        patience=OPTIMIZER_SWITCHER_PATIENCE
    )
    scheduler = optimizer.get_scheduler(total_steps=1000, warmup_steps=100)
