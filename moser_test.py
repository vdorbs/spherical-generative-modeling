from argparse import ArgumentParser
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.pyplot import figure, show
from numpy import pi
from spherical_generative_modeling.distributions import Uniform
from spherical_generative_modeling.models.moser_flows import SphereFluxField, MoserSphereVectorField
from torch import acos, atan2, arange, cat, float64, linspace, log, maximum, minimum, no_grad, sin, stack, tensor, zeros
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint


parser = ArgumentParser()
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()

num_samples = 10000
data = zeros(0, 3, dtype=float64)
num_new_samples = num_samples
while len(data) < num_samples:
    new_data = Uniform().sample((num_new_samples,))
    new_longs = atan2(new_data[:, 1], new_data[:, 0])
    new_colats = acos(new_data[:, 2])
    is_acceptable = sin(2 * new_longs) * sin(4 * new_colats) < 0.
    data = cat([data, new_data[is_acceptable]])
data = data[:num_samples].to(args.device)

# longs = atan2(data[:, 1], data[:, 0])
# colats = acos(data[:, 2])
#
# fig = figure(figsize=(8, 4), tight_layout=True)
#
# ax = fig.add_subplot(1, 2, 1, projection='3d')
# ax.scatter(*data.T, s=1)
# ax.view_init(azim=45)
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.set_zlim(-1, 1)
#
# ax = fig.add_subplot(1, 2, 2)
# ax.scatter(longs, colats, s=1)
# ax.axis('equal')
# ax.set_xlim(-pi, pi)
# ax.set_ylim(0, pi)
# ax.grid()
#
# show()

training_data = TensorDataset(data[:5000], data[:5000])
validation_data = TensorDataset(data[5000:], data[5000:])
uniform_data = Uniform().sample((50000,)).to(args.device)

model = SphereFluxField(num_hidden_layers=3, hidden_dim=32).to(args.device)
base_distribution = Uniform()
vector_field = MoserSphereVectorField(base_distribution, model)

optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
epsilon = tensor(1e-5, dtype=float64)
lambda_pos = 10

num_epochs = 1000
for epoch in range(num_epochs):
    for step, (batch, _) in enumerate(DataLoader(training_data, batch_size=256, shuffle=True)):
        densities = vector_field.density(batch, tensor(0., dtype=float64))
        negative_log_likelihood = -log(maximum(densities, epsilon)).mean()
        positivity_densities = vector_field.density(uniform_data, tensor(0., dtype=float64))
        positivity_penalty = (epsilon - minimum(positivity_densities, epsilon)).mean()
        loss = negative_log_likelihood + lambda_pos * positivity_penalty
        # print(epoch, step, negative_log_likelihood.item(), positivity_penalty.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with no_grad():
        all_negative_log_likelihoods = []
        for step, (batch, _) in enumerate(DataLoader(validation_data, batch_size=5000)):
            densities = vector_field.density(batch, tensor(0., dtype=float64))
            negative_log_likelihood = -log(maximum(densities, epsilon)).mean()
            all_negative_log_likelihoods.append(negative_log_likelihood)
        negative_log_likelihood = tensor(all_negative_log_likelihoods).mean()

        positivity_densities = vector_field.density(uniform_data, tensor(0., dtype=float64))
        positivity_penalties = epsilon - minimum(positivity_densities, epsilon)
        loss = negative_log_likelihood + lambda_pos * positivity_penalties.mean()

        print(epoch, negative_log_likelihood.item(), positivity_penalties.max().item(), positivity_penalties.mean().item())

        if (epoch + 1) % 100 == 0:
            densities = vector_field.density(uniform_data, tensor(0., dtype=float64))
            longs = atan2(uniform_data[:, 1], uniform_data[:, 0])
            colats = acos(uniform_data[:, 2])

            fig = figure(figsize=(8, 6))
            ax = fig.add_subplot(1, 1, 1)
            scatter = ax.scatter(longs, colats, s=1, c=densities, cmap='jet')
            fig.colorbar(scatter)
            ax.axis('equal')
            ax.set_xlim(-pi, pi)
            ax.set_ylim(0, pi)
            ax.grid()

            show()

with no_grad():
    noise = base_distribution.sample((5000,)).to(args.device)
    longs = atan2(noise[:, 1], noise[:, 0]).cpu()
    colats = acos(noise[:, 2]).cpu()
    ts = linspace(0, 1, 200 + 1, dtype=float64, device=args.device).flip((0,))
    trajectories = odeint(vector_field, noise, ts, rtol=1e-5, atol=1e-7)

    fig = figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    scatter = ax.scatter(longs, colats, s=1)
    ax.axis('equal')
    ax.set_xlim(-pi, pi)
    ax.set_ylim(0, pi)
    ax.grid()

    def update(frame):
        longs = atan2(trajectories[frame, :, 1], trajectories[frame, :, 0]).cpu()
        colats = acos(trajectories[frame, :, 2]).cpu()
        scatter.set_offsets(stack([longs, colats], dim=-1))
        return scatter,

    frames = arange(len(ts))
    anim = FuncAnimation(fig, update, frames, interval=int(10 / (len(ts) - 1) * 1000))
    anim.save('output.gif', writer=PillowWriter(fps=20))
