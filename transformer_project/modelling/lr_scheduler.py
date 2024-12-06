class LR_Scheduler:
    def __init__(self, optimizer, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.last_lr = 0

        def step(self) -> float:
            """Perform a step."""
            result = (self.d_model**-0.5) * min(
                self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5)
            )
            self.step_num += 1
            self.last_lr = result
            return result

        def get_lr(self) -> float:
            """Return last computed learning rate."""
            return self.last_lr
