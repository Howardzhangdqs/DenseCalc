import torch
from torch.optim import AdamW
from lion_pytorch import Lion
from transformers import get_cosine_schedule_with_warmup
from EasyObj import BetterDict


class AutoOptimizerSwitcher:
    """AutoOptimizerSwitcher dynamically switches between optimizers based on a monitored metric.

    This class initializes with the Lion optimizer and switches to AdamW if the monitored
    metric does not improve for a specified number of consecutive evaluations (patience).
    It is useful for adapting the optimization strategy during training to potentially achieve
    better performance.

    Attributes:
        model (torch.nn.Module): The model to optimize.
        lr (float): Learning rate for the optimizer. Defaults to 1e-3.
        weight_decay (float): Weight decay coefficient. Defaults to 1e-2.
        patience (int): Number of consecutive non-improving steps before switching optimizers. Defaults to 5.
        best_loss (float, optional): The best loss value observed. Defaults to None.
        counter (int): Counter for consecutive non-improving steps.
        current_optimizer (str): Name of the currently active optimizer.
        optimizer (torch.optim.Optimizer): The current optimizer instance.

    Methods:
        step(loss: float):
            Updates the best loss and switches optimizer if the loss has not improved for
            a number of steps equal to patience.

        switch_optimizer():
            Switches the optimizer from Lion to AdamW and resets the counter.

        zero_grad():
            Clears the gradients of all optimized parameters.

        backward(loss: torch.Tensor):
            Performs backpropagation on the loss.

        step_optimizer():
            Updates the model parameters based on the current optimizer.

    Example usage:
        ```
        import torch.nn as nn

        # Define a simple model
        model = nn.Linear(10, 1)

        # Initialize the optimizer switcher
        optimizer_switcher = AutoOptimizerSwitcher(model)

        # Dummy training loop
        for epoch in range(10):

            # Perform a training step
            optimizer_switcher.zero_grad()
            output = model(torch.randn(10))
            loss = torch.nn.functional.mse_loss(output, torch.randn(1))
            optimizer_switcher.backward(loss)
            optimizer_switcher.step_optimizer()

            # Update the optimizer based on the loss
            optimizer_switcher.step(loss.item())

            print(f"Epoch {epoch + 1}, Loss: {loss.item()}, Current Optimizer: {optimizer_switcher.current_optimizer}")
        ```
    """

    def __init__(self, model, lr=1e-3, weight_decay=1e-2, patience=10):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.best_loss = None
        self.counter = 0
        self.current_optimizer = 'lion'
        self.optimizer = [
            Lion(self.model.parameters(), lr=self.lr / 4, weight_decay=self.weight_decay),
            AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay),
        ]
        self.optimizer_name = ['Lion', 'AdamW']
        self.optimizer_index = 0

    def update_loss(self, loss):
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.counter = 0
                self.switch_optimizer()

    def switch_optimizer(self):
        self.optimizer_index += 1
        if self.optimizer_index >= len(self.optimizer):
            self.optimizer_index = 0
        print(f"Switched optimizer to {self.optimizer_name[self.optimizer_index]}")

    def zero_grad(self):
        self.optimizer[self.optimizer_index].zero_grad()

    def backward(self, loss):
        loss.backward()

    def step(self):
        self.optimizer[self.optimizer_index].step()

    @property
    def param_groups(self):
        return self.optimizer[self.optimizer_index].param_groups

    def get_scheduler(self, total_steps, warmup_steps):
        """
        Cosine annealing learning rate scheduler with warmup.
        """

        class SchedulerWrapper:
            def __init__(inner_self):
                inner_self.schedulers = [
                    get_cosine_schedule_with_warmup(
                        optimizer=optimizer,
                        num_warmup_steps=warmup_steps,
                        num_training_steps=total_steps,
                    ) for optimizer in self.optimizer
                ]

            def step(self):
                for scheduler in self.schedulers:
                    scheduler.step()

        return SchedulerWrapper()
