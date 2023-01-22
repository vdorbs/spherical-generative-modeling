from torch import cat, float64, ones, Tensor, zeros_like
from torch.autograd import grad, set_grad_enabled
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

        normalized_x = x / norm(x, dim=-1, keepdim=True)
        t = t * ones(x.shape[:-1]).to(x)
        v = self.model(cat([normalized_x, t.unsqueeze(-1)], dim=-1))
        # Remove component in normal direction
        v_tangent = v - (normalized_x * v).sum(dim=-1).unsqueeze(-1) * normalized_x

        assert (normalized_x * v_tangent).sum(dim=-1).abs().max() < 1e-12
        return v_tangent

    def divergence(self, x: Tensor, t: Tensor) -> Tensor:
        """ Compute divergence of vector field after projecting onto sphere

        :param x: (batch_dims, 3) tensor of evaluation points
        :param t: 0-dim tensor specifying time
        :return: (batch_dims,) tensor of divergences
        """

        assert x.shape[-1] == 3
        assert x.dtype == float64
        assert len(t.shape) == 0

        normlized_x = x / norm(x, dim=-1, keepdim=True)

        with set_grad_enabled(True):
            normalized_x_with_grad = normlized_x.clone().requires_grad_(True)
            F = self(t, normalized_x_with_grad)
            div_F = zeros_like(x[..., 0])

            for i in range(3):
                div_F += grad(F[..., i].sum(), normalized_x_with_grad, create_graph=True)[0][..., i]

        return div_F

class AugmentedSphereVectorField(Module):
    def __init__(self, vector_field: SphereVectorField):
        Module.__init__(self)
        self.vector_field = vector_field

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        assert z.shape[-1] == 4
        assert z.dtype == float64
        assert len(t.shape) == 0

        x = z[..., :-1]
        x_dot = self.vector_field(t, x)
        log_p_dot = -self.vector_field.divergence(x, t)
        return cat([x_dot, log_p_dot.unsqueeze(-1)], dim=-1)
