from numpy import log, pi
from spherical_generative_modeling.models.continuous_normalizing_flows import ContinuousNormalizingFlow, SphereVectorField
from torch import cat, diff, float64, linspace, ones, randn, Size, Tensor, tensor, zeros
from torch.distributions import Distribution
from torch.linalg import cross, norm
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


class Uniform(Distribution):
    def __init__(self):
        Distribution.__init__(self, validate_args=False)

    def rsample(self, sample_shape: Size = Size()) -> Tensor:
        sample_shape = sample_shape + Size([3])
        normal_samples = randn(sample_shape, dtype=float64)
        return normal_samples / norm(normal_samples, dim=-1, keepdim=True)

    def log_prob(self, value: Tensor) -> Tensor:
        return -log(4 * pi) * ones(value.shape[:-1], dtype=float64)


class TestVectorField(Module):
    def forward(self, t: Tensor, x: Tensor):
        n = tensor([0., 0., 1.], dtype=float64)
        reshaped_n = n.reshape(Size(ones(len(x.shape) - 1, dtype=int)) + n.shape)
        return cross(reshaped_n, x)


base_distribution = Uniform()
model = SphereVectorField(num_hidden_layers=3, hidden_dim=32)
# model = TestVectorField()

# cnf = ContinuousNormalizingFlow(model, base_distribution, t_max=tensor(10., dtype=float64))
# data = base_distribution.sample((1000, 3))
# noise = cnf.normalize(data, verbose=True)
# reconstructed_data = cnf.generate(noise, verbose=True)
# print(norm(data - reconstructed_data, dim=-1).max().item())

ts = linspace(0, 10, 200 + 1, dtype=float64)
data = base_distribution.sample((1000, 3))
cnf = ContinuousNormalizingFlow(model, base_distribution, t_max=ts[-1])
normalizing_trajectory = cnf.normalize(data, ts, verbose=True)
assert (normalizing_trajectory[0] == data).all()
noise = normalizing_trajectory[-1]
generating_trajectory = cnf.generate(noise, ts, verbose=True)
assert (generating_trajectory[-1] == noise).all()
reconstructed_data = generating_trajectory[0]
print(norm(data - reconstructed_data, dim=-1).max().item())
print(norm(normalizing_trajectory - generating_trajectory, dim=-1).max().item())

base_distribution = Uniform()
model = SphereVectorField(num_hidden_layers=3, hidden_dim=32)
cnf = ContinuousNormalizingFlow(model, base_distribution)

num_samples = 1000
data = zeros(0, 3, dtype=float64)
num_new_samples = num_samples
while len(data) < num_samples:
    new_samples = base_distribution.sample((num_new_samples,))
    is_acceptable = new_samples[:, -1] >= 0.
    data = cat([data, new_samples[is_acceptable]], dim=0)
    num_new_samples *= 2
data = data[:num_samples]

batch_size = 200
data_loader = DataLoader(TensorDataset(data, data), batch_size=batch_size, shuffle=True)
optimizer = Adam(model.parameters(), lr=1e-2, maximize=True)
num_epochs = 10
for epoch in range(num_epochs):
    for step, (batch, _) in enumerate(data_loader):
        print(epoch, step)
        total_log_prob = cnf.log_prob(batch).mean()
        optimizer.zero_grad()
        total_log_prob.backward()
        optimizer.step()

    total_log_prob = cnf.log_prob(data, enable_grad=False).mean()
    print(epoch, total_log_prob.item())
