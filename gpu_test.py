import numpy
import torch
from numpy import pi
from spherical_generative_modeling.models.continuous_normalizing_flows import ContinuousNormalizingFlow, SphereVectorField
from torch import cat, float64, load, ones, randn, Size, Tensor
from torch.distributions import Distribution
from torch.linalg import norm
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
print(device)

res = load('data/data_stanford_bunny.pt')
sphere_vertices = res['sphere_vertices']
faces = res['faces']
data_face_idxs = res['data_face_idxs']
data_barycentric_coords = res['data_barycentric_coords']
sphere_data = (data_barycentric_coords.unsqueeze(-1) * sphere_vertices[faces[data_face_idxs]]).sum(dim=-2)
sphere_data /= norm(sphere_data, dim=-1, keepdim=True)

model = SphereVectorField(num_hidden_layers=3, hidden_dim=32).to(device)

class Uniform(Distribution):
    def __init__(self):
        Distribution.__init__(self, validate_args=False)
        self.device = device

    def rsample(self, sample_shape: Size = Size()) -> Tensor:
        sample_shape = sample_shape + Size([3])
        normal_samples = randn(sample_shape, dtype=float64)
        return normal_samples / norm(normal_samples, dim=-1, keepdim=True)

    def log_prob(self, value: Tensor) -> Tensor:
        return -numpy.log(4 * pi) * ones(value.shape[:-1], dtype=float64, device=value.device)

base_distribution = Uniform()
cnf = ContinuousNormalizingFlow(model, base_distribution)

training_data = sphere_data[:5000]
validation_data = sphere_data[5000:]
data_loader = DataLoader(TensorDataset(training_data, training_data), batch_size=256, shuffle=True)
validation_data_loader = DataLoader(TensorDataset(validation_data, validation_data), batch_size=256, shuffle=True)
optimizer = Adam(model.parameters(), lr=1e-2, maximize=True)

for epoch in range(1):
    for step, (batch, _) in enumerate(data_loader):
        batch = batch.to(device)
        total_log_prob = cnf.log_prob(batch).mean()
        print(epoch, step, total_log_prob.item())

        optimizer.zero_grad()
        total_log_prob.backward()
        optimizer.step()

    all_generated_sphere_samples = []
    for step, (batch, _) in enumerate(validation_data_loader):
        batch = batch.to(device)
        generated_sphere_samples = cnf.generate(batch)
        all_generated_sphere_samples.append(generated_sphere_samples.cpu())
        print(step)

    generated_sphere_samples = cat(all_generated_sphere_samples)

