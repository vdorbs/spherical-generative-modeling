from numpy import arange, arccos, array, concatenate, cos, diff, exp, float64, identity, log, logical_and, maximum, minimum, ones_like, pi, sin, sort, stack, tan, unique, zeros, zeros_like, dot
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
  

def check_length(pos0: NDArray[float], 
                 pos1: NDArray[float], 
                 target_length: float, 
                 epsilon: float =1e-6):
    """Check that the length between `pos0` and `pos1` is within `epsilon` error of the `target_length` (via assertion).
    
    :param pos0: (num_vertices,) first vertex position
    :param pos1: (num_vertices,) second vertex position
    :param target_length: what the length of the edge between `pos0` and `pos1` should be
    : returns: None
    """
    actual_length = norm(pos0 - pos1)
    assert norm(actual_length - target_length) < epsilon, f"bad length. actual: {actual_length}, target: {target_length}"


def check_angle(vec0: NDArray[float], 
                vec1: NDArray[float], 
                target_angle: float, 
                epsilon: float = 1e-6):
    """Check that the length between `vec0` and `vec1` is within `epsilon` error of `target_angle` via assertion.
    
    :param vec0: (num_vertices,) first vector
    :param vec1: (num_vertices,) second vector
    :param target_angle: what the angle between `vec0` and `vec1` should be.
    : returns: None
    """
    actual_angle = arccos(dot(vec0 / norm(vec0), vec1 / norm(vec1)))
    assert abs(actual_angle - target_angle) < epsilon, f"bad angle. actual: {actual_angle}, target: {target_angle}"

def get_angle(l_left: float, 
              l_opp: float, 
              l_right: float):
    """Get the angle between the edges with lengths `l_left` and `l_right`.
    
    :param l_left: the length of the vector on the "left" side of the angle
    :param l_right: the length of the vector on the "right" side of the angle
    :param l_opp: the length of the vector on the opposite side of the angle (so its facing it)
    : returns: (float) the angle
    """
    angle = arccos((-l_opp**2 + l_left**2 + l_right**2) / (2*l_left*l_right))
    return angle % pi

def get_rotation_matrix(rotation_angle: float) -> NDArray[float]:
    """Get the matrix that rotates a vector of length 2 by `rotation_angle` (counterclockwise).
    
    :rotation_angle: the angle to rotate by
    : returns: (2, 2) the rotation matrix
    """
    return array([[cos(rotation_angle), -sin(rotation_angle)], [sin(rotation_angle), cos(rotation_angle)]])


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
    
    """
    Step 1: Compute angles
    """
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
    
    """
    Step 2: Create mappings for the mesh layout.
    
    An "edge" here is a tuple `(i, j)` where `i` and `j` are indices
    (i.e. the indices are elements of `kept_faces`)
    """
    edge_to_opposite = {} # map the edge (i, j) to k, where k is the index opposite to the edge (i, j)
    edge_to_length = {} # map the edge (i, j) to l, where l is the length of the edge of (i, j)
    edge_to_angle = {} # map the edge (i, j) to the angle opposite to the edge between thos vertices

    for face_idx, (cur_face, cur_edge_lengths, cur_angles) in enumerate(zip(faces, edge_lengths, angles)):
        for (i0, i1, i2), edge_length in zip([(0, 1, 2), (1, 2, 0), (2, 0, 1)], cur_edge_lengths):
            
            edge = (cur_face[i0], cur_face[i1])
            assert edge not in edge_to_opposite, "edge was already used"
            edge_to_opposite[edge] = cur_face[i2]
            edge_to_length[edge] = edge_length
            edge_to_angle[edge] = cur_angles[i2]
            
            
    """
    Step 3: layout the mesh

    Procedure:

    create the first edge and add it to `queue`
    while `queue` is not empty:
        1) edge <- first element in `queue` (pop it from `queue)
        2) start, end <- edge (`start` and `end` are the two vertices in `edge`)
        3) find `opp` (the opposite vertex) from `start` and `end` using the input edge lengths
           This is identical to the procedure in the original layout code 
           i.e. we compute the new position in two ways and then take the mean
        4) add the edges (opp, start) and (end, opp) to the front of `queue`
    """
    vertex_positions = zeros(shape=(num_vertices, 2))

    # create the first edge
    i0, i1, _ = faces[starting_idx]
    vertex_positions[i0] = [0., 0.]
    vertex_positions[i1] = [0., edge_to_length[(i0, i1)]]

    # initialize `queue` and `seen`
    queue = []
    queue.append((i0, i1))
    queue.append((i1, i0))
    seen = set()

    i = 0
    while queue:

        edge = queue.pop(0)

        if (edge in edge_to_opposite) and (edge not in seen):

            i += 1

            # retrieve vertex indices
            opp = edge_to_opposite[edge] # this is the (index of) the unpositioned vertex (opposite to `start` and `end`)
            start, end = edge # these are the indices for the already positioned vertices

            # retrieve vector lengths
            l_start = edge_to_length[(opp, start)]
            l_end = edge_to_length[(end, opp)]

            # retrieve positions
            pos_start = vertex_positions[start]
            pos_end = vertex_positions[end]

            # retrieve angles
            angle_start = edge_to_angle[(end, opp)]
            angle_end = edge_to_angle[(opp, start)]

            # get position using vector from `start` to `opp`
            edge_vector = pos_end - pos_start
            edge_vector /= norm(edge_vector)
            rotation_matrix = get_rotation_matrix(angle_start)
            new_pos0 =  pos_start + rotation_matrix @ (l_start*edge_vector)

            # get position using vector from `end` to `opp`
            edge_vector = pos_start - pos_end
            edge_vector /= norm(edge_vector)
            rotation_matrix = get_rotation_matrix(angle_end).T # rotate it clockwise (for rotation matrices, the transpose is the inverse)
            new_pos1 = pos_end + rotation_matrix @ (l_end * edge_vector)

            # Set the position of `opp` (the new position)
            # pos_opp = (new_pos1 + new_pos0) / 2
            pos_opp = new_pos0
            vertex_positions[opp] = pos_opp

            # Report
            l_target = edge_to_length[edge] # target length between start and end
            l_actual = norm(vertex_positions[end] - vertex_positions[start]) # actual length between start and end
            if verbose and i % 1000 == 0: print(f'extrinsically_layout: iteration {i}, actual length: {l_actual}, target length: {l_target}')

            # Check angles and length vs targets (via assertions)
            # assert norm(new_pos1 - new_pos0) < 1e-6, f"Positions don't match: {new_pos1}, {new_pos0}"
            # check_angle(pos_opp - pos_start, pos_end - pos_start, angle_start)
            # check_length(pos_start, pos_end, l_target)
            check_length(pos_start, pos_opp, l_start)
            # check_length(pos_end, pos_opp, l_end)

            # Update `queue`: add new edges (via depth-first)
            queue.insert(0, (start, opp))
            queue.insert(0, (opp, end))

            # Update `seen` (edges we have already seen)
            seen.add((opp, start))
            seen.add((end, opp))
            seen.add(edge)
            
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
