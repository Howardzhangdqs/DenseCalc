import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from typing import Any, Tuple
from rich.progress import Progress
from tqdm import tqdm
from optimizer import AutoOptimizerSwitcher
# from utils import int_length


def int_length(n: int) -> int:
    """
    Get the length of an integer
    """
    return len(str(n))


def train(
    model: nn.Module,
    train_loader: data.DataLoader,
    optimizer: AutoOptimizerSwitcher | optim.Optimizer,
    criterion: Any,
    scheduler: Any = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    epoch: Tuple[int, int] = None,
) -> None:
    model.train()
    model.to(device)
    progress_bar = tqdm(
        train_loader,
        desc=f"Train" if epoch is None else f"Train [{epoch[0]: {int_length(epoch[1])}}/{epoch[1]}]",
        # leave=False
    )

    train_loss = 0

    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        progress_bar.set_postfix(loss=f"{loss.item():.5f}", lr=f"{optimizer.param_groups[0]['lr']:.5f}")

        train_loss += loss.item()
    progress_bar.close()

    if "update_loss" in dir(optimizer):
        optimizer.update_loss(loss.item())

    print(f"Loss: {train_loss / len(train_loader)}")


def validate(
    model: nn.Module,
    val_loader: data.DataLoader,
    criterion: Any,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    epoch: Tuple[int, int] = None,
) -> None:
    model.eval()
    model.to(device)
    progress_bar = tqdm(
        val_loader,
        desc=f"Validate" if epoch is None else f"Validate [{epoch[0]: {int_length(epoch[1])}}/{epoch[1]}]",
        # leave=False
    )

    val_loss = 0

    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        progress_bar.set_postfix(loss=f"{loss.item():.5f}")

        val_loss += loss.item()
    progress_bar.close()

    print(f"Loss: {val_loss / len(val_loader)}")


if __name__ == "__main__":
    from optimizer import AutoOptimizerSwitcher
    model = nn.Sequential(nn.Linear(1, 10), nn.ReLU(), nn.Linear(10, 1))
    optimizer = AutoOptimizerSwitcher(model, patience=20)
    criterion = nn.MSELoss()

    # sin wave data
    train_data = torch.linspace(0, 100, 100).unsqueeze(1)
    train_target = torch.sin(train_data)

    # add noise
    train_target += torch.randn_like(train_target) * 0.1

    dataset = data.TensorDataset(train_data, train_target)

    train_loader = data.DataLoader(dataset, batch_size=2, shuffle=True)
    # train(model, train_loader, optimizer, criterion, epoch=(100, 10))

    for epoch in range(100):
        train(model, train_loader, optimizer, criterion, epoch=(epoch + 1, 100))
