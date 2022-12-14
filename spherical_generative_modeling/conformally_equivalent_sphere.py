from torch import arange, arccos, cat, diff, float64, int64, IntTensor, minimum, ones, ones_like, Size, sqrt, Tensor, tensor
from torch.linalg import cross, norm
from typing import Tuple


class ConformallyEquivalentSphere:
    def __init__(self, vertices: Tensor, faces: Tensor):
        """ Utilities for spherical interpolation and flows

        :param vertices: (num_vertices, 3) tensor of spherical vertex positions
        :param faces: (num_faces, 3) tensor of vertices comprising each face
        """

        assert len(vertices.shape) == 2
        assert vertices.shape[-1] == 3
        assert vertices.dtype == float64
        assert len(faces.shape) == 2
        assert faces.shape[-1] == 3
        assert faces.dtype == int64

        num_vertices = len(vertices)

        assert faces.min() == 0
        assert faces.max() == num_vertices - 1
        assert (norm(vertices, dim=-1) - 1.).abs().max() < 1e-12

        vertices_by_face = vertices[faces]
        vertices_by_face_extra = cat([vertices_by_face, vertices_by_face[:, :2, :]], dim=1)

        # Compute logs and orientations
        bases = vertices_by_face_extra[:, :-2, :]
        targets = vertices_by_face_extra[:, 1:-1, :]
        opposites = vertices_by_face_extra[:, 2:, :]

        sphere_logs = self.log(bases, targets)
        hemisphere_normals = cross(bases, sphere_logs) # Rotate pi/2 CCW
        orientations = (opposites * hemisphere_normals).sum(dim=-1).sign()

        self.vertices = vertices
        self.faces = faces
        self.hemisphere_normals = hemisphere_normals
        self.orientations = orientations

    def log(self, base: Tensor, target: Tensor) -> Tensor:
        """ Compute spherical logarithms

        :param base: (batch_dims, 3) tensor of base points
        :param target: (batch_dims, 3) tensor of log arguments
        :return: (batch_dims, 3) tensor of tangent vectors at base
        """

        assert base.shape == target.shape
        assert base.shape[-1] == 3
        assert base.dtype == float64
        assert target.dtype == float64
        assert (norm(base, dim=-1) - 1.).abs().max() < 1e-12
        assert (norm(target, dim=-1) - 1.).abs().max() < 1e-12

        cos_angle = (base * target).sum(dim=-1)
        is_over_1 = cos_angle >= 1.
        cos_angle = minimum(cos_angle, ones_like(cos_angle))
        angle = arccos(cos_angle)
        sin_angle = sqrt(1 - (cos_angle ** 2))
        ratio = ones_like(angle)
        ratio[~is_over_1] = angle[~is_over_1] / sin_angle[~is_over_1]
        unnormalized_projection = target - cos_angle.unsqueeze(-1) * base
        sphere_log = ratio.unsqueeze(-1) * unnormalized_projection

        assert (base * sphere_log).sum(dim=-1).abs().max() < 1e-12
        return sphere_log

    def locate(self, query_point: Tensor) -> Tuple[IntTensor, Tensor]:
        """ Determine query point memberships in spherical triangles

        :param query_point: (batch_dims, 3) tensor of query points
        :return: (batch_dims,) tensor of face indices and (batch_dims, 3) tensor of barycentric coordinates
        """

        assert query_point.shape[-1] == 3
        assert query_point.dtype == float64
        assert (norm(query_point, dim=-1) - 1.).abs().max() < 1e-12

        # Find containing spherical triangles through hemisphere tests
        batch_dims = query_point.shape[:-1]
        reshape_dims = Size(ones(len(batch_dims), dtype=int))
        reshaped_query_point = query_point.reshape(batch_dims + Size([1, 1, 3]))
        reshaped_hemisphere_normals = self.hemisphere_normals.reshape(reshape_dims + self.hemisphere_normals.shape)
        reshaped_orientations = self.orientations.reshape(reshape_dims + self.orientations.shape)
        hemisphere_memberships = (reshaped_query_point * reshaped_hemisphere_normals).sum(dim=-1).sign() == reshaped_orientations
        spherical_triangle_memberships = hemisphere_memberships.all(dim=-1)
        assert (spherical_triangle_memberships.sum(dim=-1) == 1).all()

        num_faces = len(self.faces)
        face_idxs = arange(num_faces, dtype=int64)
        reshaped_face_idxs = face_idxs.reshape(reshape_dims + Size([num_faces]))
        spherical_triangle_idx = (spherical_triangle_memberships.to(int64) * reshaped_face_idxs).sum(dim=-1)
        assert spherical_triangle_idx.min() >= 0
        assert spherical_triangle_idx.max() <= num_faces - 1

        # Project query points onto Euclidean triangles
        containing_vertices = self.vertices[self.faces[spherical_triangle_idx]]
        containing_vertices_extra = cat([containing_vertices, containing_vertices[..., :1, :]], dim=-2)
        containing_euclidean_edge_vectors = diff(containing_vertices_extra, dim=-2)
        containing_euclidean_normal = cross(containing_euclidean_edge_vectors[..., 0, :], -containing_euclidean_edge_vectors[..., -1, :])
        euclidean_projection_magnitude = (containing_vertices[..., 0, :] * containing_euclidean_normal).sum(dim=-1) / (query_point * containing_euclidean_normal).sum(dim=-1)
        euclidean_projection = euclidean_projection_magnitude.unsqueeze(-1) * query_point

        # Compute barycentric coordinates
        containing_euclidean_area = norm(containing_euclidean_normal, dim=-1) / 2
        containing_euclidean_diffs = euclidean_projection.unsqueeze(-2) - containing_vertices
        containing_euclidean_subareas = norm(cross(containing_euclidean_edge_vectors, containing_euclidean_diffs), dim=-1) / 2
        barycentric_coords = containing_euclidean_subareas[..., tensor([1, 2, 0])] / containing_euclidean_area.unsqueeze(-1)
        assert (barycentric_coords.sum(dim=-1) - 1.).abs().max() < 1e-12

        return spherical_triangle_idx, barycentric_coords
