from torch import exp, float64, Tensor, zeros_like
from torch.autograd import grad, set_grad_enabled
from torch.distributions import Distribution
from torch.linalg import norm
from torch.nn import Linear, Module, Sequential, Tanh


class SphereFluxField(Module):
    def __init__(self, num_hidden_layers: int, hidden_dim: int):
        """ Parametrize a flux field on the unit sphere by a neural network

        :param num_hidden_layers: number of hidden layers (not including input or output layers)
        :param hidden_dim: dimension of each hidden layer
        """
        Module.__init__(self)
        layers = [Linear(3, hidden_dim, dtype=float64), Tanh()]
        for _ in range(num_hidden_layers):
            layers += [Linear(hidden_dim, hidden_dim, dtype=float64), Tanh()]
        layers += [Linear(hidden_dim, 3, dtype=float64)]
        self.model = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """ Compute tangent vectors given by vector field after projecting onto sphere

        :param x: (batch_dims, 3) tensor of evaluation points
        :return: (batch_dims, 3) tensor of tangent vectors on the unit sphere
        """

        assert x.shape[-1] == 3
        assert x.dtype == float64

        normalized_x = x / norm(x, dim=-1, keepdim=True)
        v = self.model(normalized_x)
        # Remove component in normal direction
        v_tangent = v - (normalized_x * v).sum(dim=-1).unsqueeze(-1) * normalized_x

        assert (normalized_x * v_tangent).sum(dim=-1).abs().max() < 1e-12
        return v_tangent

    def divergence(self, x: Tensor) -> Tensor:
        """ Compute divergence of vector field after projecting onto sphere

        :param x: (batch_dims, 3) tensor of evaluation points
        :return: (batch_dims,) tensor of divergences
        """

        assert x.shape[-1] == 3
        assert x.dtype == float64

        x /= norm(x, dim=-1, keepdim=True)

        with set_grad_enabled(True):
            x_with_grad = x.clone().requires_grad_(True)
            F = self(x_with_grad)
            div_F = zeros_like(x[..., 0])

            for i in range(3):
                div_F += grad(F[..., i].sum(), x_with_grad, create_graph=True)[0][..., i]

        return div_F


class MoserSphereVectorField(Module):
    def __init__(self, base_distribution: Distribution, sphere_flux_field: SphereFluxField):
        """ Construct a vector field on unit sphere derived from flux field as in Moser Flows

        :param base_distribution: tractable noise distribution on sphere
        :param sphere_flux_field: SphereFluxField object pushing density away from base distribution
        """

        Module.__init__(self)
        self.base_distribution = base_distribution
        self.sphere_flux_field = sphere_flux_field

    def density(self, x: Tensor, t: Tensor) -> Tensor:
        """ Compute time-varying probability density at samples

        :param x: (batch_dims, 3) tensor of evaluation points
        :param t: 0-dim tensor specifying time
        :return: (batch_dims,) tensor of probability densities
        """

        assert x.shape[-1] == 3
        assert x.dtype == float64

        return exp(self.base_distribution.log_prob(x)) + (1. - t) * self.sphere_flux_field.divergence(x)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """ Compute tangent vectors given by vector field after projecting onto sphere

        :param t: 0-dim tensor specifying time
        :param x: (batch_dims, 3) tensor of evaluation points
        :return: (batch_dims, 3) tensor of tangent vectors on the unit sphere
        """

        assert x.shape[-1] == 3
        assert x.dtype == float64

        return self.sphere_flux_field(x) / self.density(x, t).unsqueeze(-1)
