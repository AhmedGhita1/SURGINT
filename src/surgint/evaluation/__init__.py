from collections.abc import Callable

import torch


MetricsFn = Callable[[torch.nn.Module], dict[str, float]]
