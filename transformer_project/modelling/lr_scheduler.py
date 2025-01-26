class LR_Scheduler:
    def __init__(self, optimizer, d_model: int, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 1
        self.last_lr = 0

    def calculate_lr(self, step: int):
        return (self.d_model**-0.5) * min(
            step ** (-0.5), step * self.warmup_steps ** (-1.5)
        )

    def get_lr_schedule(self, num_steps: int):
        return [self.calculate_lr(step) for step in range(1, num_steps + 1)]

    def step(self):
        lr = self.calculate_lr(self.step_num)
        self.step_num += 1
        self.last_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_lr(self):
        return self.last_lr
