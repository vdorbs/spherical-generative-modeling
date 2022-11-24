from argparse import ArgumentParser
from matplotlib.pyplot import figure, show
from numpy import array, concatenate, diff, exp, stack
from numpy.linalg import norm
from potpourri3d import read_mesh
from trimesh import Trimesh

from spherical_generative_modeling import ConformallyEquivalentSphere, parametrize


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

total_log_factors, new_vertices = parametrize(vertices, faces, mesh.area_faces, flatten_max_iters=400, flatten_step_size=0.1, verbose=True)

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

assert abs(new_edge_lengths - exp(total_log_factor_averages) * edge_lengths).max() < 1e-12

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


from matplotlib.animation import FuncAnimation
from numpy import pi
from spherical_generative_modeling.models.continuous_normalizing_flows import SphereVectorFieldTangentRepresentation
from torch import arange, cat, cos, float64, int64, linspace, logical_and, meshgrid, no_grad, ones, randn, sin, Size, stack, Tensor, tensor, zeros, zeros_like
from torch.linalg import cross, norm
from torch.nn import Module
from torchdiffeq import odeint, odeint_event


new_vertices = tensor(new_vertices, dtype=float64)
faces = tensor(faces, dtype=int64)
total_log_factors = tensor(total_log_factors, dtype=float64)
sphere = ConformallyEquivalentSphere(new_vertices, faces)

query_points = randn(10, 3, dtype=float64)
query_points /= norm(query_points, dim=-1, keepdim=True)
spherical_triangle_idxs, _ = sphere.locate(query_points)

ax = figure(figsize=(6, 6)).add_subplot(1, 1, 1, projection='3d')
ax.plot_trisurf(*new_vertices.T, triangles=faces, alpha=0.1)
ax.scatter(*query_points.T, c='C1')
for face_idx in spherical_triangle_idxs:
    spherical_triangle_vertices = new_vertices[faces[face_idx]]
    ax.scatter(*spherical_triangle_vertices.T, c='C2')
ax.view_init(azim=45)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel('$x$', fontsize=16)
ax.set_ylabel('$y$', fontsize=16)
ax.set_zlabel('$z$', fontsize=16)
show()

# Interpolate log conformal factors onto spherical grid
bandwidth = 32
longs = 2 * pi * arange(2 * bandwidth, dtype=float64) / (2 * bandwidth)
colats = pi * (2 * arange(2 * bandwidth, dtype=float64) + 1) / (4 * bandwidth)
long_grid, colat_grid = meshgrid(longs, colats, indexing='ij')
all_longs = long_grid.reshape(-1)
all_colats = colat_grid.reshape(-1)
all_carts = cat([sin(all_colats).unsqueeze(-1) * stack([cos(all_longs), sin(all_longs)], dim=-1), cos(all_colats).unsqueeze(-1)], dim=-1)
spherical_triangle_idxs, barycentric_coords = sphere.locate(all_carts)
interpolated_total_log_factors = (total_log_factors[faces[spherical_triangle_idxs]] * barycentric_coords).sum(dim=-1)

fig = figure(figsize=(12, 6), tight_layout=True)

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(*new_vertices.T, s=5, c=total_log_factors, cmap='jet')
ax.view_init(azim=45)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel('$x$', fontsize=16)
ax.set_ylabel('$y$', fontsize=16)
ax.set_zlabel('$z$', fontsize=16)

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter(*all_carts.T, s=5, c=interpolated_total_log_factors, cmap='jet')
ax.view_init(azim=45)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel('$x$', fontsize=16)
ax.set_ylabel('$y$', fontsize=16)
ax.set_zlabel('$z$', fontsize=16)

show()


class TestVectorField(Module):
    def forward(self, t: Tensor, x: Tensor):
        n = tensor([0., 0., 1.], dtype=float64)
        reshaped_n = n.reshape(Size(ones(len(x.shape) - 1, dtype=int)) + n.shape)
        return cross(reshaped_n, x)


model = TestVectorField()

x_0 = randn(1000, 3, dtype=float64)
x_0 /= norm(x_0, dim=-1, keepdim=True)
xs = zeros(Size([0]) + x_0.shape)
ts = linspace(0, 10, 200 + 1, dtype=float64)

x_curr = x_0.clone()
t_curr = ts[0]
while t_curr < ts[-1]:
    local_model = SphereVectorFieldTangentRepresentation(model, x_curr, ts[-1])
    v_curr = zeros_like(x_curr)

    with no_grad():
        t_next, _ = odeint_event(local_model, v_curr, t_curr, event_fn=local_model.event_fn)
        is_between_events = logical_and(ts >= t_curr, ts <= t_next)
        ts_between_events = ts[is_between_events]
        vs_between_events = odeint(local_model, v_curr, ts_between_events)
        xs_between_events = local_model.to_manifold(vs_between_events)

    xs = cat([xs, xs_between_events[:-1]])
    print(t_next.item(), xs.shape, ts_between_events[-1].item())
    x_curr = xs_between_events[-1]
    t_curr = ts_between_events[-1]

xs = cat([xs, xs_between_events[-1:]])

fig = figure(figsize=(6, 6), tight_layout=True)
ax = fig.add_subplot(1, 1, 1, projection='3d')

ax.scatter(*xs[0].T, s=1, alpha=0.5)
scatter = ax.scatter(*xs[0].T)

ax.view_init(azim=45)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel('$x$', fontsize=16)
ax.set_ylabel('$y$', fontsize=16)
ax.set_zlabel('$z$', fontsize=16)

def update(frame):
    scatter._offsets3d = tuple(xs[frame].numpy().T)
    return scatter,

frames = arange(len(ts))
anim = FuncAnimation(fig, update, frames, interval=int(1000 * 0.05))
anim.save('output.gif', writer='imagemagik')
