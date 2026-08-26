from collections.abc import Callable

import torch

# the contract between a scoring function and whatever consumes its numbers
MetricsFn = Callable[[torch.nn.Module], dict[str, float]]
