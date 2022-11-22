from numpy import arange, arccos, array, concatenate, cos, diff, exp, float64, identity, log, logical_and, maximum, minimum, ones_like, pi, sin, sort, stack, tan, unique, zeros, zeros_like
from numpy.linalg import norm, solve
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, dia_matrix
from scipy.sparse.linalg import spsolve
from typing import Tuple


def intrinsically_normalize_boundary(
        vertices: NDArray[float64],
        faces: NDArray[float64],
        is_boundary: NDArray[bool],
        removed_vertex: NDArray[float64]
) -> Tuple[NDArray[float64], NDArray[float64]]:
    """ Make boundary vertices equidistant from removed vertex

    :param vertices: (num_vertices, 3) array of vertex positions
    :param faces: (num_faces, 3)
    :param is_boundary: (num_vertices,) array of boundary truth values
    :param removed_vertex: (3,) position of removed vertex
    :return: (num_vertices,) array of log conformal factors and (num_faces, 3) array of new edge lengths
    """

    num_vertices = len(vertices)

    assert len(vertices.shape) == 2
    assert vertices.shape[-1] == 3
    assert len(vertices) == len(is_boundary)
    assert len(faces.shape) == 2
    assert faces.shape[-1] == 3
    assert faces.min() == 0
    assert faces.max() == num_vertices - 1
    assert len(removed_vertex) == 3
    assert is_boundary.sum() > 0

    removed_edge_lengths = norm(vertices[is_boundary] - removed_vertex, axis=-1)
    log_factors = zeros(num_vertices)
    log_factors[is_boundary] = 2 * log(removed_edge_lengths.mean() / removed_edge_lengths)

    faces_extra = concatenate([faces, faces[:, :1]], axis=-1)
    log_factor_columns = stack([log_factors[indices] for indices in faces_extra.T], axis=-1)
    log_factor_averages = (log_factor_columns[:, :-1] + log_factor_columns[:, 1:]) / 2

    vertex_columns = stack([vertices[indices] for indices in faces_extra.T], axis=-2)
    vertex_diffs = diff(vertex_columns, axis=-2)
    edge_lengths = norm(vertex_diffs, axis=-1)

    new_edge_lengths = exp(log_factor_averages) * edge_lengths
    return log_factors, new_edge_lengths


def intrinsically_flatten(
        faces: NDArray[int],
        is_boundary: NDArray[bool],
        edge_lengths: NDArray[float64],
        max_iters: int = 200,
        step_size: float = 1.,
        verbose: bool = False
) -> Tuple[NDArray[float64], NDArray[float64]]:
    """Compute log conformal factors to flatten a topological disk, following CETM

    :param faces: (num_faces, 3) array of vertices comprising each face
    :param is_boundary: (num_vertices,) array of boundary truth values
    :param edge_lengths: (num_faces, 3) array of edge lengths for each face
    :param max_iters: maximum number of optimization iterations
    :param step_size: multiplier on optimization step size
    :param verbose: print optimization details
    :return: (num_vertices,) array of flat log conformal factors and (num_faces, 3) array of new edge lengths
    """

    num_vertices = len(is_boundary)

    assert faces.shape == edge_lengths.shape
    assert len(faces.shape) == 2
    assert faces.shape[-1] == 3
    assert faces.min() == 0
    assert faces.max() == num_vertices - 1
    assert is_boundary.sum() > 0
    assert max_iters >= 0
    assert step_size >= 0.

    log_factors = zeros(num_vertices)
    faces_extra = concatenate([faces, faces[:, :1]], axis=-1)

    for iteration in range(max_iters):
        # Compute new edge lengths from log conformal factors
        log_factor_columns = stack([log_factors[indices] for indices in faces_extra.T], axis=-1)
        log_factor_averages = (log_factor_columns[:, :-1] + log_factor_columns[:, 1:]) / 2
        new_edge_lengths = exp(log_factor_averages) * edge_lengths

        # Compute new interior angles in kij order
        new_edge_lengths_extra = concatenate([new_edge_lengths, new_edge_lengths[:, :2]], axis=-1)
        l_ijs = new_edge_lengths_extra[:, :3]
        l_jks = new_edge_lengths_extra[:, 1:4]
        l_kis = new_edge_lengths_extra[:, 2:]
        # Law of cosines: l_ijs ** 2 = l_jks ** 2 + l_kis ** 2 - 2 * l_jks * l_kis * cos(alpha_ks)
        cos_new_angles = (l_jks ** 2 + l_kis ** 2 - l_ijs ** 2) / (2 * l_jks * l_kis)
        # Keep in interval [-1, 1]
        cos_new_angles = maximum(minimum(cos_new_angles, ones_like(cos_new_angles)), -ones_like(cos_new_angles))
        new_angles = arccos(cos_new_angles)

        # Compute new angle sums around vertices
        new_angle_sums = zeros(num_vertices)
        faces_kij = concatenate([faces[:, -1:], faces[:, :2]], axis=-1)
        for angle_triple, face in zip(new_angles, faces_kij):
            new_angle_sums[face] += angle_triple

        # Compute new cotans (for Laplacian)
        new_tans = tan(new_angles)
        new_is_small = abs(new_tans) < 1e-6
        new_cotans = zeros_like(new_tans)
        new_cotans[~new_is_small] = 1 / new_tans[~new_is_small]

        # Flatten new data (column major order)
        new_is = faces_extra[:, :-1].flatten('F')
        new_js = faces_extra[:, 1:].flatten('F')
        new_cotans = new_cotans.flatten('F')

        # Compute new cotan Laplacian
        new_off_diag_half_lap = coo_matrix((new_cotans, (new_is, new_js)), shape=(num_vertices, num_vertices)).tocsr()
        new_off_diag_lap = new_off_diag_half_lap + new_off_diag_half_lap.T
        new_diag_lap = array(new_off_diag_lap.sum(axis=-1))[:, 0]
        new_diag_lap = dia_matrix((new_diag_lap + 1e-6, 0), shape=(num_vertices, num_vertices))  # Bump spectrum away from 0
        new_laplacian = (new_diag_lap - new_off_diag_lap) / 2

        # Optimization step
        new_gradient = 2 * pi - new_angle_sums[~is_boundary]
        new_hessian = new_laplacian[~is_boundary][:, ~is_boundary]
        new_step = spsolve(new_hessian, new_gradient)
        log_factors[~is_boundary] -= step_size * new_step

        if verbose:
            print(iteration, norm(new_gradient), abs(new_gradient).max(), abs(new_hessian).max(), norm(new_step))

    # Compute new edge lengths from log conformal factors
    log_factor_columns = stack([log_factors[indices] for indices in faces_extra.T], axis=-1)
    log_factor_averages = (log_factor_columns[:, :-1] + log_factor_columns[:, 1:]) / 2
    new_edge_lengths = exp(log_factor_averages) * edge_lengths

    return log_factors, new_edge_lengths


def extrinsically_layout(
        num_vertices: int,
        faces: NDArray[int],
        edge_lengths: NDArray[float64],
        starting_idx: int = 0,
        verbose: bool = False
) -> NDArray[float64]:
    """ Compute vertex positions for flat mesh from edge lengths

    :param num_vertices: number of vertices
    :param faces: (num_faces, 3) array of vertices comprising each face
    :param edge_lengths: (num_faces, 3) array of edge lengths for each face
    :param starting_idx: face index from which to start search
    :param verbose: print layout details
    :return: (num_vertices, 2) array of vertex positions in the plane
    """

    assert faces.shape == edge_lengths.shape
    assert len(faces.shape) == 2
    assert faces.shape[-1] == 3
    assert faces.min() == 0
    assert faces.max() == num_vertices - 1
    assert edge_lengths.min() >= 0.
    assert starting_idx >= 0
    assert starting_idx < len(faces)

    num_faces = len(faces)
    vertex_positions = zeros((num_vertices, 2))
    is_positioned = zeros(num_vertices, dtype=bool)
    is_visited = zeros(num_faces, dtype=bool)

    # Compute interior angles
    edge_lengths_extra = concatenate([edge_lengths, edge_lengths[:, :2]], axis=-1)
    l_ijs = edge_lengths_extra[:, :3]
    l_jks = edge_lengths_extra[:, 1:4]
    l_kis = edge_lengths_extra[:, 2:]
    # Law of cosines: l_ijs ** 2 = l_jks ** 2 + l_kis ** 2 - 2 * l_jks * l_kis * cos(alpha_ks)
    cos_angles = (l_jks ** 2 + l_kis ** 2 - l_ijs ** 2) / (2 * l_jks * l_kis)
    # Keep in interval [-1, 1]
    cos_angles = maximum(minimum(cos_angles, ones_like(cos_angles)), -ones_like(cos_angles))
    angles = arccos(cos_angles)
    # Switch to ijk order
    angles = angles[:, array([1, 2, 0])]

    # Initial face
    face = faces[starting_idx]
    i, j, k = face
    l_ij, l_jk, l_ki = edge_lengths[starting_idx]
    alpha_i, alpha_j, alpha_k = angles[starting_idx]

    # Place vertex i at the origin, vertex j directly up, and vertex k in CCW order
    # Assumes starting face has no edges on the boundary
    vertex_positions[j] = array([0., l_ij])
    vertex_positions[k] = l_ki * array([-sin(alpha_i), cos(alpha_i)])
    is_positioned[face] = True
    is_visited[starting_idx] = True

    # Find neighboring faces
    shared_vertex_mask = (faces.reshape(-1, 3, 1) == face.reshape(1, 1, 3)).any(axis=1)
    shares_edge = shared_vertex_mask.sum(axis=-1) == 2
    assert shares_edge.sum() == 3  # Assumes starting face has no edges on the boundary
    adjacent_idxs = arange(num_faces)[shares_edge]

    # Recursively place remaining faces
    next_frontier = adjacent_idxs
    while len(next_frontier) > 0:
        if verbose:
            print(is_positioned.sum(), num_vertices, is_visited.sum(), num_faces, len(next_frontier))

        frontier = next_frontier
        next_frontier = array([], dtype=int)
        for face_idx in frontier:
            if ~is_visited[face_idx]:
                face = faces[face_idx]
                local_is_positioned = is_positioned[face]
                # A face is only in the frontier if a neighboring face had all vertices positioned
                assert local_is_positioned.sum() >= 2

                # A face can be unvisited but have all vertices positioned if all neighbors were previously visited
                if local_is_positioned.sum() == 2:
                    local_vertex_positions = vertex_positions[face]  # Exactly one of these is not positioned
                    local_edge_lengths = edge_lengths[face_idx]
                    local_angles = angles[face_idx]

                    # Classify vertices
                    idx_orders = array([[1, 2, 0], [2, 0, 1], [0, 1, 2]])
                    start, end, unpositioned = idx_orders[~local_is_positioned][0]
                    
                    edge_vector = local_vertex_positions[end] - local_vertex_positions[start]
                    edge_vector /= norm(edge_vector)
                    rotation_angle = local_angles[start]
                    rotation_matrix = array([
                        [cos(rotation_angle), -sin(rotation_angle)],
                        [sin(rotation_angle), cos(rotation_angle)]
                    ])
                    new_position = local_vertex_positions[start] + rotation_matrix @ (local_edge_lengths[unpositioned] * edge_vector)

                    vertex_positions[face[unpositioned]] = new_position
                    is_positioned[face[unpositioned]] = True

                # Find neighboring faces
                shared_vertex_mask = (faces.reshape(-1, 3, 1) == face.reshape(1, 1, 3)).any(axis=1)
                shares_edge = shared_vertex_mask.sum(axis=-1) == 2
                assert shares_edge.sum() > 0
                assert shares_edge.sum() <= 3
                explorable = logical_and(shares_edge, ~is_visited)
                adjacent_idxs = arange(num_faces)[explorable]

                next_frontier = concatenate([next_frontier, adjacent_idxs])
                is_visited[face_idx] = True

    assert is_visited.all()
    return vertex_positions


def stereographically_project(planar_vertices: NDArray[float64]) -> Tuple[NDArray[float64], NDArray[float64]]:
    """ Stereographically project planar vertices onto unit sphere through north pole

    :param planar_vertices: (num_vertices, 2) array of vertex positions
    :return: (num_vertices,) array of log conformal factors and (num_vertices, 3) array of vertex positions on sphere
    """

    assert len(planar_vertices.shape) == 2
    assert planar_vertices.shape[-1] == 2

    xs, ys = planar_vertices.T
    r_squareds = (xs ** 2) + (ys ** 2)
    stereo_vertex_positions = stack([2 * xs, 2 * ys, r_squareds - 1], axis=1) / (r_squareds.reshape(-1, 1) + 1)
    log_factors = log(2 / (r_squareds + 1))

    return log_factors, stereo_vertex_positions


def mobius_normalize(
        spherical_vertices: NDArray[float64],
        faces: NDArray[int],
        original_areas: NDArray[float64],
        max_iters: int = 10,
        verbose: bool = False
) -> Tuple[NDArray[float64], NDArray[float64]]:
    """ Apply Möbius transformations to move area-weighted center of mass to the origin, following Möbius Registration

    :param spherical_vertices: (num_vertices, 3) array of vertex positions
    :param faces: (num_faces, 3) array of vertices comprising each face
    :param original_areas: (num_faces,) array of face areas in original mesh
    :param max_iters: maximum number of Möbius transformations
    :param verbose: print details of Möbius transformations
    :return: (num_vertices,) array of log conformal factors and (num_vertices, 3) array of vertex positions
    """

    num_vertices = len(spherical_vertices)

    assert len(spherical_vertices.shape) == 2
    assert spherical_vertices.shape[-1] == 3
    assert len(faces) == len(original_areas)
    assert faces.min() == 0
    assert faces.max() == num_vertices - 1
    assert original_areas.min() >= 0.
    assert max_iters >= 0

    assert abs(norm(spherical_vertices, axis=-1) - 1).max() < 1e-12

    area_proportions = original_areas / original_areas.sum()
    log_factors = zeros(num_vertices)

    for iteration in range(max_iters):
        # Compute area-weighted center of mass of face centers
        face_centers = spherical_vertices[faces].mean(axis=-2)
        face_centers /= norm(face_centers, axis=-1, keepdims=True)
        center_of_mass = face_centers.T @ area_proportions

        # Compute spherical inversion center
        face_center_outer_products = face_centers.reshape(-1, 3, 1) * face_centers.reshape(-1, 1, 3)
        center_of_mass_jac = 2 * (identity(3) - (area_proportions.reshape(-1, 1, 1) * face_center_outer_products).sum(axis=0))
        inversion_center = -solve(center_of_mass_jac, center_of_mass)

        # Line search to put center inside sphere
        first_divisions = 0
        while norm(inversion_center) > 1:
            inversion_center /= 2.
            first_divisions += 1

        # Apply transformation
        translated_vertices = spherical_vertices + inversion_center
        reflected_vertices = translated_vertices / (norm(translated_vertices, axis=-1, keepdims=True) ** 2)
        next_vertices = (1 - (norm(inversion_center) ** 2)) * reflected_vertices + inversion_center

        next_face_centers = next_vertices[faces].mean(axis=-2)
        next_face_centers /= norm(next_face_centers, axis=-1, keepdims=True)
        next_center_of_mass = next_face_centers.T @ area_proportions

        # Line search for improvement
        second_divisions = 0
        while norm(next_center_of_mass) > norm(center_of_mass):
            inversion_center /= 2.
            second_divisions += 1

            translated_vertices = spherical_vertices + inversion_center
            reflected_vertices = translated_vertices / (norm(translated_vertices, axis=-1, keepdims=True) ** 2)
            next_vertices = (1 - (norm(inversion_center) ** 2)) * reflected_vertices + inversion_center

            next_face_centers = next_vertices[faces].mean(axis=-2)
            next_face_centers /= norm(next_face_centers, axis=-1, keepdims=True)
            next_center_of_mass = next_face_centers.T @ area_proportions

        log_factors += log((1 - norm(inversion_center) ** 2) / (norm(spherical_vertices + inversion_center, axis=1) ** 2))
        spherical_vertices = next_vertices

        if verbose:
            print(iteration, first_divisions, second_divisions, next_center_of_mass)

    return log_factors, spherical_vertices


def parametrize(
        vertices: NDArray[float64],
        faces: NDArray[float64],
        face_areas: NDArray[float64],
        flatten_max_iters: int = 200,
        flatten_step_size: float = 1.,
        layout_starting_idx: int = 0,
        mobius_max_iters: int = 10,
        verbose: bool = False
) -> Tuple[NDArray[float64], NDArray[float64]]:
    """ Compute centered conformal spherical parametrization, unique up to rotation

    :param vertices: (num_vertices, 3) array of vertex positions
    :param faces: (num_faces, 3) array of vertices comprising each face
    :param face_areas: (num_faces,) array of face areas
    :param flatten_max_iters: maximum number of optimization iterations for intrinsic flattening
    :param flatten_step_size: multiplier on optimization step size for intrinsic flattening
    :param layout_starting_idx: face index from which to start search for extrinsic layout
    :param mobius_max_iters: maximum number of Möbius transformations
    :param verbose: print all details
    :return: (num_vertices,) array of log conformal factors and (num_vertices, 3) array of spherical centered vertex positions
    """

    num_vertices = len(vertices)

    assert len(vertices.shape) == 2
    assert vertices.shape[-1] == 3
    assert len(faces.shape) == 2
    assert faces.shape[-1] == 3
    assert faces.min() == 0
    assert faces.max() == num_vertices - 1
    assert len(face_areas) == len(faces)
    assert faces.min() >= 0.
    assert flatten_max_iters >= 0
    assert flatten_step_size >= 0.
    assert layout_starting_idx >= 0
    assert layout_starting_idx < len(faces)
    assert mobius_max_iters >= 0

    # Remove first vertex
    removed_vertex = vertices[0]
    kept_vertices = vertices[1:]

    # Remove faces containing first vertex
    is_contained = (faces == 0).any(axis=-1)
    removed_faces = faces[is_contained]
    kept_faces = faces[~is_contained]

    # Correct indexing for kept faces
    kept_faces -= 1

    # Log new boundary vertices
    boundary_vertices = sort(unique(removed_faces.flatten()))[1:] - 1
    is_boundary = (arange(len(kept_vertices)).reshape(-1, 1) == boundary_vertices.reshape(1, -1)).any(axis=1)

    # Compute flat discrete metric
    normalized_log_factors, normalized_edge_lengths = intrinsically_normalize_boundary(kept_vertices, kept_faces, is_boundary, removed_vertex)
    flat_log_factors, flat_edge_lengths = intrinsically_flatten(kept_faces, is_boundary, normalized_edge_lengths, max_iters=flatten_max_iters, step_size=flatten_step_size, verbose=verbose)

    # Compute planar vertex layout
    num_vertices = len(kept_vertices)
    planar_vertices = extrinsically_layout(num_vertices, kept_faces, flat_edge_lengths, starting_idx=layout_starting_idx, verbose=verbose)

    # Stereographically project to unit sphere
    stereo_log_factors, stereo_vertices = stereographically_project(planar_vertices)

    # Reinsert removed vertex
    total_log_factors = normalized_log_factors + flat_log_factors + stereo_log_factors
    original_removed_edge_lengths = norm(kept_vertices[is_boundary] - removed_vertex, axis=-1)
    removed_log_factor_versions = -total_log_factors[is_boundary] + 2 * log(norm(stereo_vertices[is_boundary] - array([0., 0., 1.]), axis=-1) / original_removed_edge_lengths)
    removed_log_factor = removed_log_factor_versions.mean()
    total_log_factors = concatenate([array([removed_log_factor]), total_log_factors])

    stereo_vertices = concatenate([array([[0., 0., 1.]]), stereo_vertices])

    # Apply Möbius normalizations
    mobius_log_factors, mobius_vertices = mobius_normalize(stereo_vertices, faces, face_areas, max_iters=mobius_max_iters, verbose=verbose)

    total_log_factors += mobius_log_factors
    new_vertices = mobius_vertices
    return total_log_factors, new_vertices
