from matplotlib.pyplot import figure, show
from numpy import pi
from spherical_generative_modeling import align
from torch import acos, atan2, float64, load, randn, tensor
from torch.linalg import matrix_exp


res = load('data/data_stanford_bunny.pt')
sphere_vertices = res['sphere_vertices']
faces = res['faces']
log_factors = res['log_factors']

omega = randn(4, dtype=float64)
Omega = tensor([
    [0., -omega[3], omega[2]],
    [omega[3], 0., -omega[1]],
    [-omega[2], omega[1], 0.]
], dtype=float64)
R = matrix_exp(Omega)
rotated_sphere_vertices = (R @ sphere_vertices.T).T

fig = figure(figsize=(16, 6))

ax = fig.add_subplot(1, 2, 1)
ax.scatter(atan2(sphere_vertices[:, 1], sphere_vertices[:, 0]), acos(sphere_vertices[:, 2]), s=0.1, c=log_factors, cmap='jet')
ax.axis('equal')
ax.set_xlim(-pi, pi)
ax.set_ylim(0, pi)
ax.grid()

ax = fig.add_subplot(1, 2, 2)
ax.scatter(atan2(rotated_sphere_vertices[:, 1], rotated_sphere_vertices[:, 0]), acos(rotated_sphere_vertices[:, 2]), s=0.1, c=log_factors, cmap='jet')
ax.axis('equal')
ax.set_xlim(-pi, pi)
ax.set_ylim(0, pi)
ax.grid()

show()

best_rotation = align(8, sphere_vertices, faces, log_factors, rotated_sphere_vertices, faces, log_factors)
reconstructed_sphere_vertices = (best_rotation @ rotated_sphere_vertices.T).T

fig = figure(figsize=(16, 6))

ax = fig.add_subplot(1, 2, 1)
ax.scatter(atan2(sphere_vertices[:, 1], sphere_vertices[:, 0]), acos(sphere_vertices[:, 2]), s=0.1, c=log_factors, cmap='jet')
ax.axis('equal')
ax.set_xlim(-pi, pi)
ax.set_ylim(0, pi)
ax.grid()

ax = fig.add_subplot(1, 2, 2)
ax.scatter(atan2(reconstructed_sphere_vertices[:, 1], reconstructed_sphere_vertices[:, 0]), acos(reconstructed_sphere_vertices[:, 2]), s=0.1, c=log_factors, cmap='jet')
ax.axis('equal')
ax.set_xlim(-pi, pi)
ax.set_ylim(0, pi)
ax.grid()

show()