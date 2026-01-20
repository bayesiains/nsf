import math

from torch import nn, optim
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmUpLR(_LRScheduler):
    def __init__(self, optimizer, warm_up_epochs, total_epochs, eta_min=0, last_epoch=-1):
        self.warm_up_epochs = warm_up_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warm_up_epochs:
            return [base_lr * (self.last_epoch / self.warm_up_epochs)
                    for base_lr in self.base_lrs]
        else:
            frac_epochs = (
                    (self.last_epoch - self.warm_up_epochs)
                    / (self.total_epochs - self.warm_up_epochs)
            )
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * frac_epochs)) / 2
                    for base_lr in self.base_lrs]


def main():
    model = nn.Linear(5, 3)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    warm_up_steps = 0
    total_steps = 100
    scheduler = CosineAnnealingWarmUpLR(optimizer, warm_up_steps, total_steps)
    lrs = []
    for _ in range(total_steps):
        optimizer.zero_grad()
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_lr()[0])
    print(lrs)


if __name__ == '__main__':
    main()
