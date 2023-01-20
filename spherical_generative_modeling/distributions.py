from torch import float64, randn, Size, Tensor, zeros
from torch.distributions import Distribution
from torch.linalg import norm


class Uniform(Distribution):
    def __init__(self):
        Distribution.__init__(self, validate_args=False)

    def rsample(self, sample_shape: Size = Size()) -> Tensor:
        sample_shape = sample_shape + Size([3])
        normal_samples = randn(sample_shape, dtype=float64)
        return normal_samples / norm(normal_samples, dim=-1, keepdim=True)

    def log_prob(self, value: Tensor) -> Tensor:
        return zeros(value.shape[:-1], dtype=float64).to(value)
