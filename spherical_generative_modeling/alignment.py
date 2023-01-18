from .conformally_equivalent_sphere import ConformallyEquivalentSphere
from math import factorial
import numpy as np
from torch import arange, cat, complex128, cos, diag, float64, IntTensor, ones, sin, sqrt, stack, Tensor, tensor, zeros
from torch.fft import ifft2, ifftshift
from torch.linalg import norm
from ts2kit.ts2kit import dltWeightsDH, FTSHT, gridDH


def R_3(angle: Tensor) -> Tensor:
    """ Rotate about z axis

    :param angle: angle of rotation
    :return: (3, 3) rotation matrix
    """

    assert len(angle.shape) == 0

    c = cos(angle)
    s = sin(angle)
    return tensor([
        [c, -s, 0.],
        [s, c, 0.],
        [0., 0., 1.]
    ], dtype=float64)

def R_2(angle: Tensor) -> Tensor:
    """ Rotate about y axis

    :param angle: angle of rotation
    :return: (3, 3) rotation matrix
    """

    assert len(angle.shape) == 0

    c = cos(angle)
    s = sin(angle)
    return tensor([
        [c, 0., s],
        [0., 1., 0.],
        [-s, 0., c]
    ], dtype=float64)


def align(
        bandwidth: int,
        f_vertices: Tensor,
        f_faces: IntTensor,
        f: Tensor,
        g_vertices: Tensor,
        g_faces: IntTensor,
        g: Tensor
) -> Tensor:
    """ Compute rotation of signal g that maximizes correlation with signal f

    :param bandwidth: signal bandwidths
    :param f_vertices: (num_vertices_f, 3) tensor of sphere vertices for signal f
    :param f_faces: (num_faces_f, 3) tensor of vertex indices in each face for signal f
    :param f: (num_vertices_f,) tensor of values for signal f
    :param g_vertices: (num_vertices_g, 3) tensor of sphere vertices for signal f
    :param g_faces: (num_faces_g, 3) tensor of vertex indices in each face for signal f
    :param g: (num_vertices_g,) tensor of values for signal f
    :return: (3, 3) tensor rotating g to maximize correlation with f
    """

    assert bandwidth > 0
    assert bandwidth % 2 == 0

    assert len(f_vertices.shape) == 2
    assert len(f_faces.shape) == 2
    assert len(f.shape) == 1
    assert len(f_vertices) == len(f)

    assert len(g_vertices.shape) == 2
    assert len(g_faces.shape) == 2
    assert len(g.shape) == 1
    assert len(g_vertices) == len(g)

    assert (norm(f_vertices, dim=-1) - 1.).abs().max() < 1e-12
    assert (norm(g_vertices, dim=-1) - 1.).abs().max() < 1e-12

    long_grid, colat_grid = gridDH(bandwidth)
    all_longs = long_grid.reshape(-1)
    all_colats = colat_grid.reshape(-1)
    all_carts = cat([sin(all_colats).unsqueeze(-1) * stack([cos(all_longs), sin(all_longs)], dim=-1), cos(all_colats).unsqueeze(-1)], dim=-1)

    # Sample f and g onto grids
    face_idxs, barycentric_coords = ConformallyEquivalentSphere(f_vertices, f_faces).locate(all_carts)
    f_grids = (barycentric_coords * f[f_faces[face_idxs]]).sum(dim=-1).reshape(2 * bandwidth, 2 * bandwidth)

    face_idxs, barycentric_coords = ConformallyEquivalentSphere(g_vertices, g_faces).locate(all_carts)
    g_grids = (barycentric_coords * g[g_faces[face_idxs]]).sum(dim=-1).reshape(2 * bandwidth, 2 * bandwidth)

    # Compute fast spherical harmonic transforms of f and g as well as SO(3) Fourier transform of convolution
    s2_fsh_transformer = FTSHT(bandwidth)
    f_hats, g_hats = s2_fsh_transformer(stack([f_grids, g_grids], dim=0))
    C_hats = g_hats.T.unsqueeze(-1) @ f_hats.conj().T.unsqueeze(-2)

    weights = dltWeightsDH(bandwidth)
    longs = long_grid[:, 0]
    colats = colat_grid[0]
    alphas = longs.clone()
    betas = colats.clone()
    gammas = longs.clone()

    # Compute inverse Wigner transform
    C_inverse_wigners = zeros(2 * bandwidth, 2 * bandwidth - 1, 2 * bandwidth - 1, dtype=complex128)

    # (0, 0) case
    m = 0
    k = 0
    wigner_matrix = zeros(bandwidth, 2 * bandwidth, dtype=float64)
    wigner_matrix[0] = 1.
    wigner_matrix[1] = cos(betas)
    for l in range(1, bandwidth - 1):
        prevs = np.sqrt((l ** 2 - m ** 2) * (l ** 2 - k ** 2)) / (l * (2 * l + 1)) * wigner_matrix[l - 1]
        currs = (m * k / (l * (l + 1)) - cos(betas)) * wigner_matrix[l]
        next_coeff = np.sqrt(((l + 1) ** 2 - m ** 2) * ((l + 1) ** 2 - k ** 2)) / ((l + 1) * (2 * l + 1))
        wigner_matrix[l + 1] = -(prevs + currs) / next_coeff

    # (0, 0) check first few terms
    assert (wigner_matrix[2] - (3 * cos(betas) ** 2 - 1) / 2).abs().max() < 1e-12
    assert (wigner_matrix[3] - (-cos(betas) * (3 - 5 * cos(betas) ** 2) / 2)).abs().max() < 1e-12
    assert (wigner_matrix[4] - ((3 - 30 * cos(betas) ** 2 + 35 * cos(betas) ** 4) / 8)).abs().max() < 1e-12

    # Normalize and check inversion
    normalized_wigner_matrix = sqrt((2 * arange(bandwidth, dtype=float64) + 1) / 2).unsqueeze(-1) * wigner_matrix
    inverse_then_forward = normalized_wigner_matrix @ (weights.unsqueeze(-1) * normalized_wigner_matrix.T)
    assert (diag(inverse_then_forward) - 1).abs().max() < 1e-12
    assert norm(inverse_then_forward - diag(diag(inverse_then_forward)), 2) < 1e-12

    # Save inverse to C_inverse_wigners
    C_hat_fiber = C_hats[:, bandwidth - 1, bandwidth - 1]
    C_inverse_wigner_fiber = normalized_wigner_matrix.to(complex128).T @ C_hat_fiber
    C_inverse_wigners[:, bandwidth - 1, bandwidth - 1] = C_inverse_wigner_fiber

    # Remaining cases
    for degree in range(1, bandwidth):
        ones_vec = ones(2 * degree + 1, dtype=int)
        arange_vec = arange(-degree, degree + 1, dtype=int)
        ms = stack([cat([-degree * ones_vec, degree * ones_vec]), cat([arange_vec, arange_vec])])
        ks = stack([cat([arange_vec, arange_vec]), cat([-degree * ones_vec, degree * ones_vec])])

        for init_mode, (m_row, k_row) in enumerate(zip(ms, ks)):
            for m, k in stack([m_row, k_row], dim=-1):
                wigner_matrix = zeros(bandwidth, 2 * bandwidth, dtype=float64)

                if init_mode == 0:
                    coeff = np.sqrt(factorial(2 * degree) / (factorial(degree + k) * factorial(degree - k)))
                    wigner_matrix[degree] = coeff * (cos(betas / 2) ** (degree + m.sign() * k)) * ((m.sign() * sin(betas / 2)) ** (degree - m.sign() * k))
                else:
                    coeff = np.sqrt(factorial(2 * degree) / (factorial(degree + m) * factorial(degree - m)))
                    wigner_matrix[degree] = coeff * (cos(betas / 2) ** (degree + k.sign() * m)) * ((-k.sign() * sin(betas / 2)) ** (degree - k.sign() * m))

                for l in range(degree, bandwidth - 1):
                    prevs = np.sqrt((l ** 2 - m ** 2) * (l ** 2 - k ** 2)) / (l * (2 * l + 1)) * wigner_matrix[l - 1]
                    currs = (m * k / (l * (l + 1)) - cos(betas)) * wigner_matrix[l]
                    next_coeff = np.sqrt(((l + 1) ** 2 - m ** 2) * ((l + 1) ** 2 - k ** 2)) / ((l + 1) * (2 * l + 1))
                    wigner_matrix[l + 1] = -(prevs + currs) / next_coeff

                wigner_matrix = wigner_matrix[degree:]

                # Normalize and check inversion
                normalized_wigner_matrix = sqrt((2 * arange(degree, bandwidth, dtype=float64) + 1) / 2).unsqueeze(-1) * wigner_matrix
                inverse_then_forward = normalized_wigner_matrix @ (weights.unsqueeze(-1) * normalized_wigner_matrix.T)
                print(
                    degree,
                    m.item(),
                    k.item(),
                    (diag(inverse_then_forward) - 1).abs().max().item(),
                    norm(inverse_then_forward - diag(diag(inverse_then_forward)), 2).item()
                )
                assert (diag(inverse_then_forward) - 1).abs().max() < 1e-6
                assert norm(inverse_then_forward - diag(diag(inverse_then_forward)), 2) < 1e-4

                # Save inverse to C_inverse_wigners
                C_hat_fiber = C_hats[degree:, bandwidth - 1 + m, bandwidth - 1 + k]
                C_inverse_wigner_fiber = normalized_wigner_matrix.to(complex128).T @ C_hat_fiber
                C_inverse_wigners[:, bandwidth - 1 + m, bandwidth - 1 + k] = C_inverse_wigner_fiber

    # Inverse Fourier transform of inverse Wigner transform
    C_inverse_wigners_padded = zeros(2 * bandwidth, 2 * bandwidth, 2 * bandwidth, dtype=complex128)
    C_inverse_wigners_padded[:, 1:, 1:] = C_inverse_wigners
    Cs = ifft2(ifftshift(C_inverse_wigners_padded.conj(), dim=(-2, -1)), s=(2 * bandwidth, 2 * bandwidth), norm='forward')
    assert Cs.imag.abs().max() < 1e-12
    Cs = Cs.real

    max_beta_idx = Cs.max(dim=-1)[0].max(dim=-1)[0].argmax()
    max_alpha_idx = Cs[max_beta_idx].max(dim=-1)[0].argmax()
    max_gamma_idx = Cs[max_beta_idx, max_alpha_idx].argmax()
    max_alpha, max_beta, max_gamma = alphas[max_alpha_idx], betas[max_beta_idx], gammas[max_gamma_idx]
    max_rotation = R_3(max_gamma) @ R_2(max_beta) @ R_3(max_alpha)
    return max_rotation