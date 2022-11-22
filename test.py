from argparse import ArgumentParser
from matplotlib.pyplot import figure, show
from numpy import array, concatenate, diff, exp, stack
from numpy.linalg import norm
from potpourri3d import read_mesh
from trimesh import Trimesh

from spherical_generative_modeling import parametrize


parser = ArgumentParser()
parser.add_argument('--mesh', type=str)
args = parser.parse_args()

vertices, faces = read_mesh(args.mesh)
# Realign so -z points in +x direction
vertices = array([[-1., 1., 1.]]) * vertices[:, array([2, 0, 1])]
# Create mesh for bounding sphere
mesh = Trimesh(vertices, faces)

ax = figure(figsize=(6, 6)).add_subplot(1, 1, 1, projection='3d')
ax.plot_trisurf(*vertices.T, triangles=faces)
ax.scatter(*mesh.bounding_sphere.vertices.T, visible=False)
ax.view_init(azim=45)
ax.set_xlabel('$x$', fontsize=16)
ax.set_ylabel('$y$', fontsize=16)
ax.set_zlabel('$z$', fontsize=16)
show()

total_log_factors, new_vertices = parametrize(vertices, faces, mesh.area_faces, flatten_step_size=0.1, verbose=True)

# Conformality check
faces_extra = concatenate([faces, faces[:, :1]], axis=-1)

vertex_columns = stack([vertices[indices] for indices in faces_extra.T], axis=-2)
vertex_diffs = diff(vertex_columns, axis=-2)
edge_lengths = norm(vertex_diffs, axis=-1)

new_vertex_columns = stack([new_vertices[indices] for indices in faces_extra.T], axis=-2)
new_vertex_diffs = diff(new_vertex_columns, axis=-2)
new_edge_lengths = norm(new_vertex_diffs, axis=-1)

total_log_factor_columns = stack([total_log_factors[indices] for indices in faces_extra.T], axis=-1)
total_log_factor_averages = (total_log_factor_columns[:, :-1] + total_log_factor_columns[:, 1:]) / 2

assert abs(new_edge_lengths - exp(total_log_factor_averages) * edge_lengths).max() < 1e-6

ax = figure(figsize=(6, 6)).add_subplot(1, 1, 1, projection='3d')
ax.scatter(*new_vertices.T, s=1)
ax.plot_trisurf(*new_vertices.T, triangles=faces, alpha=0.1)
ax.view_init(azim=45)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel('$x$', fontsize=16)
ax.set_ylabel('$y$', fontsize=16)
ax.set_zlabel('$z$', fontsize=16)
show()
