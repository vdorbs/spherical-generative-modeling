from numpy import pi
from torch import cat, cos, float64, ones, sin, Size, Tensor, zeros_like
from torch.linalg import norm
from torch.nn import Linear, Module, Sequential, Tanh


class SphereVectorField(Module):
    def __init__(self, num_hidden_layers: int, hidden_dim: int):
        """ Parametrize a vector field on the unit sphere by a neural network

        :param num_hidden_layers: number of hidden layers (not including input or output layers)
        :param hidden_dim: dimension of each hidden layer
        """
        Module.__init__(self)
        layers = [Linear(3 + 1, hidden_dim, dtype=float64), Tanh()]
        for _ in range(num_hidden_layers):
            layers += [Linear(hidden_dim, hidden_dim, dtype=float64), Tanh()]
        layers += [Linear(hidden_dim, 3, dtype=float64)]
        self.model = Sequential(*layers)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """ Compute tangent vectors given by vector field

        :param t: 0-dim tensor specifying time
        :param x: (batch_dims, 3) tensor of evaluation points on the unit sphere
        :return: (batch_dims, 3) tensor of tangent vectors at evaluation points
        """

        assert len(t.shape) == 0
        assert x.shape[-1] == 3
        assert x.dtype == float64
        assert (norm(x, dim=-1) - 1.).abs().max() < 1e-12

        t = t * ones(x.shape[:-1])
        v = self.model(cat([x, t.unsqueeze(-1)]), dim=-1)
        # Remove component in normal direction
        v_tangent = v - (x * v).sum(dim=-1).unsqueeze(-1) * x

        assert (x * v_tangent).sum(dim=-1).abs().max() < 1e-12
        return v_tangent


class SphereVectorFieldTangentRepresentation(Module):
    def __init__(self, model: SphereVectorField, base_point: Tensor, t_max: Tensor, bound: float = pi / 2, tol: float = 1e-12):
        """ Represent a SphereVectorField by equivalent vector fields in tangent spaces

        :param model: SphereVectorField to represent in tangent spaces
        :param base_point: (batch_dims, 3) tensor specifying base points of tangent spaces
        :param t_max: 0-dim tensor specifying maximum integration time
        :param bound: bound on tangent vector norm to trigger chart switches
        :param tol: tolerance for exp, log, and derivative computations
        """

        assert base_point.shape[-1] == 3
        assert base_point.dtype == float64
        assert (norm(base_point, dim=-1) - 1.).abs().max() < 1e-12
        assert len(t_max.shape) == 0
        assert t_max > 0.
        assert bound > 0.
        assert bound < pi
        assert tol > 0.

        Module.__init__(self)
        self.model = model
        self.base_point = base_point
        self.t_max = t_max
        self.bound = bound
        self.tol = tol

    def forward(self, t: Tensor, v: Tensor) -> Tensor:
        """ Compute tangent vectors given by local vector field

        :param t: 0-dim tensor specifying time
        :param v: (batch_dims, 3) tensor of evaluation points in the tangent spaces
        :return: (batch_dims, 3) tensor of tangent vectors at evaluation points
        """

        assert len(t.shape) == 0
        assert v.shape == self.base_point.shape
        assert v.dtype == float64
        assert (self.base_point * v).sum(dim=-1).abs().max() < 1e-12

        # Exponentiate tangent vectors
        norm_v = norm(v, dim=-1)
        is_small = norm_v < self.tol

        exp_v = zeros_like(v)
        exp_v[is_small] = self.base_point[is_small]

        v = v[~is_small]
        norm_v = norm_v[~is_small].unsqueeze(-1)
        cos_norm_v = cos(norm_v)
        sin_norm_v = sin(norm_v)
        exp_v[~is_small] = cos(norm_v) * self.base_point[~is_small] + sin(norm_v) * v / norm_v
        assert (norm(exp_v, dim=-1) - 1).abs().max() < 1e-12

        # Compute nonlocal tangent vectors specified by vector field
        F_exp_v = self.model(t, exp_v)

        # Push tangent vectors to tangent spaces at base points
        d_log_F_exp_v = zeros_like(F_exp_v)
        d_log_F_exp_v[is_small] = F_exp_v[is_small] - (self.base_point[is_small] * F_exp_v[is_small]).sum(dim=-1).unsqueeze(-1) * self.base_point[is_small]

        F_exp_v = F_exp_v[~is_small]
        unprojected = norm_v / sin_norm_v * F_exp_v - (sin_norm_v - norm_v * cos_norm_v) / (sin_norm_v ** 3) * (self.base_point[~is_small] * F_exp_v).sum(dim=-1).unsqueeze(-1) * exp_v[~is_small]
        d_log_F_exp_v[~is_small] = unprojected - (self.base_point[~is_small] * unprojected).sum(dim=-1).unsqueeze(-1) * self.base_point[~is_small]

        assert (self.base_point * d_log_F_exp_v).sum(dim=-1).abs().max() < 1e-12
        return d_log_F_exp_v

    def event_fn(self, t: Tensor, v: Tensor) -> Tensor:
        """ Trigger events after max time or when tangent vectors exceed norm bounds

        :param t: 0-dim tensor specifying time
        :param v: (batch_dims, 3) tensor of evaluation points in the tangent spaces
        :return: flattened tensor of norm and time differences, an event is triggered when any value is nonnegative
        """

        assert len(t.shape) == 0
        assert v.shape == self.base_point.shape
        assert v.dtype == float64
        assert (self.base_point * v).sum(dim=-1).abs().max() < 1e-12

        valid_state = norm(v, dim=-1) - self.bound
        valid_time = t - self.t_max
        return cat([valid_state.flatten(), valid_time.unsqueeze(-1)])

    def to_manifold(self, v: Tensor) -> Tensor:
        """ Map tangent vectors to points on unit sphere

        :param v: (extra_dims, batch_dims, 3) tensor of tangent vectors
        :return: (extra_dims, batch_dims, 3) tensor of corresponding points on the unit sphere
        """

        shape = self.base_point.shape
        assert v.shape[-len(shape):] == self.base_point.shape
        assert v.dtype == float64
        assert (self.base_point * v).sum(dim=-1).abs().max() < 1e-12

        norm_v = norm(v, dim=-1)
        is_small = norm_v < self.tol

        exp_v = zeros_like(v)
        reshaped_base_point = self.base_point.reshape(Size(ones(len(v.shape) - len(shape), dtype=int)) + shape)
        repeated_base_point = reshaped_base_point.repeat(v.shape[:-len(shape)] + Size(ones(len(shape), dtype=int)))
        exp_v[is_small] = repeated_base_point[is_small]

        v = v[~is_small]
        norm_v = norm_v[~is_small].unsqueeze(-1)
        exp_v[~is_small] = cos(norm_v) * repeated_base_point[~is_small] + sin(norm_v) * v / norm_v

        assert (norm(exp_v, dim=-1) - 1).abs().max() < 1e-12
        return exp_v
