from argparse import ArgumentParser
from matplotlib.pyplot import figure, show
from numpy import pi
from potpourri3d import face_areas
from spherical_generative_modeling import ConformallyEquivalentSphere
from spherical_generative_modeling.distributions import Uniform
from spherical_generative_modeling.models.continuous_normalizing_flows import SphereVectorField, AugmentedSphereVectorField
from spherical_generative_modeling.models.moser_flows import MoserSphereVectorField, SphereFluxField
from torch import acos, atan2, cat, float64, linspace, load, log, maximum, minimum, multinomial, no_grad, sqrt, stack, rand, tensor
from torch.linalg import norm
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint
import wandb


parser = ArgumentParser()
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--arch', type=str, default='moser', choices=['moser', 'cnf', 'nmode'])
parser.add_argument('--mesh', type=str, default='stanford_bunny', choices=['banana', 'camera', 'cell_phone', 'elephant', 'hammer', 'knife', 'light_bulb', 'mouse', 'rubber_duck', 'stanford_bunny'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_epochs', type=int, default=40)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--log_each', type=int, default=10)
parser.add_argument('--num_hidden_layers', type=int, default=3)
parser.add_argument('--hidden_dim', type=int, default=32)
args = parser.parse_args()

wandb.init(project='experiments', entity='vdorbs', config=args)

res = load(f'data/data_{args.mesh}.pt')
mesh_vertices = res['mesh_vertices']
sphere_vertices = res['sphere_vertices']
faces = res['faces']
data_face_idxs = res['data_face_idxs']
data_barycentric_coords = res['data_barycentric_coords']
inscribed_data = (data_barycentric_coords.unsqueeze(-1) * sphere_vertices[faces[data_face_idxs]]).sum(dim=-2)
sphere_data = inscribed_data / norm(inscribed_data, dim=-1, keepdim=True)

sphere = ConformallyEquivalentSphere(sphere_vertices, faces)
mesh_areas = tensor(face_areas(mesh_vertices, faces), dtype=float64)
inscribed_areas = tensor(face_areas(sphere_vertices, faces), dtype=float64)
data_mesh_areas = mesh_areas[data_face_idxs]
data_inscribed_areas = inscribed_areas[data_face_idxs]
changes_of_area = (sphere.change_of_area(data_face_idxs, data_barycentric_coords) * data_mesh_areas / data_inscribed_areas).to(args.device)

sphere_data = sphere_data.to(args.device)
training_data = TensorDataset(sphere_data[:5000], changes_of_area[:5000])
validation_data = TensorDataset(sphere_data[5000:], changes_of_area[5000:])
noise_data = Uniform().sample((5000,)).to(args.device)

if args.arch == 'moser':
    flux_model = SphereFluxField(args.num_hidden_layers, args.hidden_dim).to(args.device)
    base_distribution = Uniform()
    vector_field = MoserSphereVectorField(base_distribution, flux_model)
    t_0 = tensor(0., dtype=float64, device=args.device)

    # Generate uniform data for positivity constraint
    uniform_face_idxs = multinomial(mesh_areas, num_samples=50000, replacement=True)
    r_1s, r_2s = rand(2, len(uniform_face_idxs))
    sqrt_r_1s = sqrt(r_1s)
    uniform_barycentric_coords = stack([1 - sqrt_r_1s, sqrt_r_1s * (1 - r_2s), r_2s * sqrt_r_1s], dim=-1)
    uniform_mesh_data = (uniform_barycentric_coords.unsqueeze(-1) * mesh_vertices[faces[uniform_face_idxs]]).sum(dim=-2)
    uniform_sphere_data = (uniform_barycentric_coords.unsqueeze(-1) * sphere_vertices[faces[uniform_face_idxs]]).sum(dim=-2)
    uniform_sphere_data /= norm(uniform_sphere_data, dim=-1, keepdim=True)
    uniform_sphere_data = uniform_sphere_data.to(args.device)
    uniform_mesh_areas = mesh_areas[uniform_face_idxs]
    uniform_inscribed_areas = inscribed_areas[uniform_face_idxs]
    uniform_changes_of_area = (sphere.change_of_area(uniform_face_idxs, uniform_barycentric_coords) * uniform_mesh_areas / uniform_inscribed_areas).to(args.device)

    epsilon = tensor(1e-5, dtype=float64, device=args.device)
    lambda_pos = 10
    optimizer = Adam(flux_model.parameters(), lr=args.lr, weight_decay=1e-5)

elif args.arch == 'cnf':
    model = SphereVectorField(args.num_hidden_layers, args.hidden_dim).to(args.device)
    augmented_model = AugmentedSphereVectorField(model)
    base_distribution = Uniform()
    ts_forward = tensor([0., 1.], dtype=float64, device=args.device)
    ts_backward = ts_forward.flip((0,))

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

else:
    raise NotImplementedError

for epoch in range(args.num_epochs):

    # TRAINING
    for step, (batch, batch_changes_of_area) in enumerate(DataLoader(training_data, batch_size=args.batch_size, shuffle=True)):
        if args.arch == 'moser':
            sphere_densities = vector_field.density(batch, t_0)
            mesh_densities = sphere_densities / batch_changes_of_area
            neg_log_likelihood = -log(maximum(mesh_densities, epsilon)).mean()

            uniform_sphere_densities = vector_field.density(uniform_sphere_data, t_0)
            uniform_mesh_densities = uniform_sphere_densities / uniform_changes_of_area
            penalty = (epsilon - minimum(uniform_mesh_densities, epsilon)).mean()

            loss = neg_log_likelihood + lambda_pos * penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elif args.arch == 'cnf':
            generated_noise = odeint(model, batch, ts_forward, rtol=1e-5, atol=1e-7)[-1]
            generated_noise_log_probs = base_distribution.log_prob(generated_noise)
            sphere_log_probs = odeint(augmented_model, cat([generated_noise, generated_noise_log_probs.unsqueeze(-1)], dim=-1), ts_backward, rtol=1e-5, atol=1e-7)[-1, ..., -1]
            mesh_log_probs = sphere_log_probs - log(batch_changes_of_area)

            neg_log_likelihood = -mesh_log_probs.mean()
            optimizer.zero_grad()
            neg_log_likelihood.backward()
            optimizer.step()

        else:
            raise NotImplementedError

    # VALIDATION AND LOGGING
    with no_grad():

        # VALIDATION
        if args.arch == 'moser':
            for batch, batch_changes_of_area in DataLoader(validation_data, batch_size=5000):
                sphere_densities = vector_field.density(batch, t_0)
                mesh_densities = sphere_densities / batch_changes_of_area
                neg_log_likelihood = -log(maximum(mesh_densities, epsilon)).mean()

            uniform_sphere_densities = vector_field.density(uniform_sphere_data, t_0)
            uniform_mesh_densities = uniform_sphere_densities / uniform_changes_of_area
            violations = epsilon - minimum(uniform_mesh_densities, epsilon)

            divergence_integral = 4 * pi * flux_model.divergence(Uniform().sample((10000,)).to(args.device)).mean()

            wandb.log({
                'neg_log_likelihood': neg_log_likelihood.item(),
                'max_pos_violation': violations.max().item(),
                'mean_pos_violation': violations.mean().item(),
                'divergence_integral': divergence_integral.item()
            })

        elif args.arch == 'cnf':
            all_neg_log_likelihoods = []
            all_reconstruction_errors = []
            all_sphere_errors = []
            for batch, batch_changes_of_area in DataLoader(validation_data, batch_size=256):
                generated_noise = odeint(model, batch, ts_forward, rtol=1e-5, atol=1e-7)[-1]
                generated_noise_log_probs = base_distribution.log_prob(generated_noise)
                dense_ts_backward = linspace(0., 1., 200 + 1, dtype=float64, device=args.device).flip((0,))
                reversed_trajectories = odeint(augmented_model, cat([generated_noise, generated_noise_log_probs.unsqueeze(-1)], dim=-1), dense_ts_backward, rtol=1e-5, atol=1e-7)
                reconstructions = reversed_trajectories[-1]

                sphere_errors = (norm(reversed_trajectories[..., :-1], dim=-1).flatten() - 1.).abs()
                all_sphere_errors.append(sphere_errors)

                reconstructed_data = reconstructions[..., :-1]
                reconstruction_errors = norm(batch - reconstructed_data, dim=-1)
                all_reconstruction_errors.append(reconstruction_errors)

                sphere_log_probs = reconstructions[..., -1]
                mesh_log_probs = sphere_log_probs - log(batch_changes_of_area)
                neg_log_likelihood = -mesh_log_probs.mean()
                all_neg_log_likelihoods.append(neg_log_likelihood)

            reconstruction_errors = cat(all_reconstruction_errors)
            sphere_errors = cat(all_sphere_errors)
            neg_log_likelihood = tensor(all_neg_log_likelihoods).mean()
            wandb.log({
                'neg_log_likelihood': neg_log_likelihood.item(),
                'max_recon_error': reconstruction_errors.max().item(),
                'mean_recon_error': reconstruction_errors.mean().item(),
                'max_validation_sphere_violation': sphere_errors.max().item(),
                'mean_validation_sphere_violation': sphere_errors.mean().item()
            })

        # LOGGING
        if (epoch + 1) % args.log_each == 0:
            if args.arch == 'moser':
                fig = figure(figsize=(12, 4), tight_layout=True)

                ax = fig.add_subplot(1, 3, 1, projection='3d')
                ax.plot_trisurf(*mesh_vertices.T, triangles=faces, alpha=0.1)
                scatter = ax.scatter(*uniform_mesh_data.T, s=0.1, c=uniform_mesh_densities.cpu(), cmap='jet', alpha=0.8)
                fig.colorbar(scatter)
                ax.view_init(azim=45)
                ax.set_xlim(-0.1, 0.1)
                ax.set_ylim(-0.1, 0.1)
                ax.set_zlim(-0.1, 0.1)

                ax = fig.add_subplot(1, 3, 2, projection='3d')
                scatter = ax.scatter(*uniform_sphere_data.cpu().T, s=0.1, c=uniform_sphere_densities.cpu(), cmap='jet', alpha=0.8)
                fig.colorbar(scatter)
                ax.view_init(azim=45)
                ax.set_xlim(-1., 1.)
                ax.set_ylim(-1., 1.)
                ax.set_zlim(-1., 1.)

                ax = fig.add_subplot(1, 3, 3)
                longs = atan2(uniform_sphere_data[:, 1], uniform_sphere_data[:, 0]).cpu()
                colats = acos(uniform_sphere_data[:, 2]).cpu()
                scatter = ax.scatter(longs, colats, s=0.1, c=uniform_sphere_densities.cpu(), cmap='jet', alpha=0.8)
                fig.colorbar(scatter)
                ax.axis('equal')
                ax.set_xlim(-pi, pi)
                ax.set_ylim(0, pi)
                ax.grid()

                wandb.log({'densities': wandb.Image(fig)})

                ts = linspace(0, 1, 200 + 1, dtype=float64, device=args.device).flip((0,))
                trajectories = odeint(vector_field, noise_data, ts, rtol=1e-5, atol=1e-7)
                sphere_violations = (norm(trajectories, dim=-1) - 1.).abs()
                wandb.log({
                    'max_sphere_violation': sphere_violations.max().item(),
                    'mean_sphere_violation': sphere_violations.mean().item()
                })

                generated_sphere_samples = trajectories[-1].cpu()
                generated_sphere_samples /= norm(generated_sphere_samples, dim=-1, keepdim=True)
                all_generated_mesh_samples = []
                for step, (batch, _) in enumerate(DataLoader(TensorDataset(generated_sphere_samples, generated_sphere_samples), batch_size=256)):
                    generated_face_idxs, generated_barycentric_coords = sphere.locate(batch)
                    generated_mesh_samples = (generated_barycentric_coords.unsqueeze(-1) * mesh_vertices[faces[generated_face_idxs]]).sum(dim=-2)
                    all_generated_mesh_samples.append(generated_mesh_samples)
                generated_mesh_samples = cat(all_generated_mesh_samples)

                fig = figure(figsize=(12, 4), tight_layout=True)

                ax = fig.add_subplot(1, 3, 1, projection='3d')
                ax.plot_trisurf(*mesh_vertices.T, triangles=faces, alpha=0.1)
                ax.scatter(*generated_mesh_samples.T, s=1, alpha=0.8)
                ax.view_init(azim=45)
                ax.set_xlim(-0.1, 0.1)
                ax.set_ylim(-0.1, 0.1)
                ax.set_zlim(-0.1, 0.1)

                ax = fig.add_subplot(1, 3, 2, projection='3d')
                ax.scatter(*generated_sphere_samples.T, s=1, alpha=0.8)
                ax.view_init(azim=45)
                ax.set_xlim(-1., 1.)
                ax.set_ylim(-1., 1.)
                ax.set_zlim(-1., 1.)

                ax = fig.add_subplot(1, 3, 3)
                longs = atan2(generated_sphere_samples[:, 1], generated_sphere_samples[:, 0])
                colats = acos(generated_sphere_samples[:, 2])
                ax.scatter(longs, colats, s=1, alpha=0.8)
                ax.axis('equal')
                ax.set_xlim(-pi, pi)
                ax.set_ylim(0, pi)
                ax.grid()

                wandb.log({'generated_samples': wandb.Image(fig)})

            elif args.arch == 'cnf':
                ts = linspace(0, 1, 200 + 1, dtype=float64, device=args.device).flip((0,))
                trajectories = odeint(model, noise_data, ts, rtol=1e-5, atol=1e-7)
                sphere_violations = (norm(trajectories, dim=-1) - 1.).abs()
                wandb.log({
                    'max_sphere_violation': sphere_violations.max().item(),
                    'mean_sphere_violation': sphere_violations.mean().item()
                })

                generated_sphere_samples = trajectories[-1].cpu()
                generated_sphere_samples /= norm(generated_sphere_samples, dim=-1, keepdim=True)
                all_generated_mesh_samples = []
                for step, (batch, _) in enumerate(DataLoader(TensorDataset(generated_sphere_samples, generated_sphere_samples), batch_size=256)):
                    generated_face_idxs, generated_barycentric_coords = sphere.locate(batch)
                    generated_mesh_samples = (generated_barycentric_coords.unsqueeze(-1) * mesh_vertices[faces[generated_face_idxs]]).sum(dim=-2)
                    all_generated_mesh_samples.append(generated_mesh_samples)
                generated_mesh_samples = cat(all_generated_mesh_samples)

                fig = figure(figsize=(12, 4), tight_layout=True)

                ax = fig.add_subplot(1, 3, 1, projection='3d')
                ax.plot_trisurf(*mesh_vertices.T, triangles=faces, alpha=0.1)
                ax.scatter(*generated_mesh_samples.T, s=1, alpha=0.8)
                ax.view_init(azim=45)
                ax.set_xlim(-0.1, 0.1)
                ax.set_ylim(-0.1, 0.1)
                ax.set_zlim(-0.1, 0.1)

                ax = fig.add_subplot(1, 3, 2, projection='3d')
                ax.scatter(*generated_sphere_samples.T, s=1, alpha=0.8)
                ax.view_init(azim=45)
                ax.set_xlim(-1., 1.)
                ax.set_ylim(-1., 1.)
                ax.set_zlim(-1., 1.)

                ax = fig.add_subplot(1, 3, 3)
                longs = atan2(generated_sphere_samples[:, 1], generated_sphere_samples[:, 0])
                colats = acos(generated_sphere_samples[:, 2])
                ax.scatter(longs, colats, s=1, alpha=0.8)
                ax.axis('equal')
                ax.set_xlim(-pi, pi)
                ax.set_ylim(0, pi)
                ax.grid()

                wandb.log({'generated_samples': wandb.Image(fig)})
