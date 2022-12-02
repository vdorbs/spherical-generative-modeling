from numpy import pi
from torch import cat, cos, float64, logical_and, no_grad, ones, sin, Size, Tensor, tensor, zeros, zeros_like
from torch.autograd import grad, set_grad_enabled
from torch.autograd.functional import jvp
from torch.distributions import Distribution
from torch.linalg import norm
from torch.nn import Linear, Module, Sequential, Tanh
from torchdiffeq import odeint, odeint_event
from typing import Optional


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
        v = self.model(cat([x, t.unsqueeze(-1)], dim=-1))
        # Remove component in normal direction
        v_tangent = v - (x * v).sum(dim=-1).unsqueeze(-1) * x

        assert (x * v_tangent).sum(dim=-1).abs().max() < 1e-12
        return v_tangent


class SphereVectorFieldTangentRepresentation(Module):
    def __init__(
            self,
            model: SphereVectorField,
            base_point: Tensor,
            bound: float = pi / 2,
            tol: float = 1e-12,
            reverse_time: bool = False,
            t_max: Optional[Tensor] = tensor(1., dtype=float64)
    ):
        """ Represent a SphereVectorField by equivalent vector fields in tangent spaces

        :param model: SphereVectorField to represent in tangent spaces
        :param base_point: (batch_dims, 3) tensor specifying base points of tangent spaces
        :param bound: bound on tangent vector norm to trigger chart switches
        :param tol: tolerance for exp, log, and derivative computations
        :param reverse_time: whether integration is performed in reverse time
        :param t_max: optional 0-dim tensor specifying maximum integration time (if reverse_time is False)
        """

        assert base_point.shape[-1] == 3
        assert base_point.dtype == float64
        assert (norm(base_point, dim=-1) - 1.).abs().max() < 1e-12
        assert len(t_max.shape) == 0
        assert bound > 0.
        assert bound < pi
        assert tol > 0.

        if reverse_time:
            t_max = None
        else:
            assert t_max is not None
            assert len(t_max.shape) == 0
            assert t_max > 0.

        Module.__init__(self)
        self.model = model
        self.base_point = base_point
        self.bound = bound
        self.tol = tol
        self.reverse_time = reverse_time
        self.t_max = t_max

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
        if self.reverse_time:
            valid_time = -t
        else:
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


class AugmentedSphereVectorFieldTangentRepresentation(Module):
    def __init__(self, local_model: SphereVectorFieldTangentRepresentation):
        """ Augment a SphereVectorFieldTangentRepresentation with log density dynamics

        :param local_model: SphereVectorFieldTangentRepresentation to be augmented
        """

        Module.__init__(self)
        self.local_model = local_model

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        """ Compute tangent vectors given by local vector field as well as changes in log densities

        :param t: 0-dim tensor specifying time
        :param z: (batch_dims, 4) tensor of evaluation points in the tangent spaces as well as log densities
        :return: (batch_dims, 4) tensor of tangent vectors at evaluation points as well as changes in log densities
        """

        assert len(t.shape) == 0
        assert z[..., :-1].shape == self.local_model.base_point.shape
        assert z.dtype == float64

        v = z[..., :-1]
        log_p = z[..., -1]

        assert (self.local_model.base_point * v).sum(dim=-1).abs().max() < 1e-12

        exp_v = self.local_model.to_manifold(v)
        d_log_F_exp_v = self.local_model(t, v)

        with set_grad_enabled(True):
            exp_v_with_grad = exp_v.clone().requires_grad_(True)
            F_exp_v = self.local_model.model(t, exp_v_with_grad)
            log_p_dot = zeros_like(log_p)

            # Euclidean divergence
            for i in range(3):
                dF_i_dx_i = grad(F_exp_v[..., i].sum(), exp_v_with_grad, create_graph=True)[0][..., i]
                log_p_dot -= dF_i_dx_i

            # Spherical divergence correction
            res = jvp(lambda x: self.local_model.model(t, x), exp_v_with_grad, exp_v, create_graph=True)
            log_p_dot += (exp_v * res[1]).sum(dim=-1)

        return cat([d_log_F_exp_v, log_p_dot.unsqueeze(-1)], dim=-1)

    def event_fn(self, t: Tensor, z: Tensor) -> Tensor:
        """ Trigger events after max time or when tangent vectors exceed norm bounds

        :param t: 0-dim tensor specifying time
        :param z: (batch_dims, 4) tensor of evaluation points in the tangent spaces as well as log densities
        :return: flattened tensor of norm and time differences, an event is triggered when any value is nonnegative
        """

        assert z.shape[-1] == 4
        return self.local_model.event_fn(t, z[..., :-1])


class ContinuousNormalizingFlow:
    def __init__(
            self,
            model: SphereVectorField,
            base_distribution: Distribution,
            t_max: Tensor = tensor(1., dtype=float64)
    ):
        """ Run a continuous normalizing flow as in Neural Manifold ODE

        :param model: vector field with learnable parameters
        :param base_distribution: tractable noise distribution
        :param t_max: integration max time
        """

        assert len(t_max.shape) == 0
        assert t_max > 0

        self.model = model
        self.base_distribution = base_distribution
        self.t_max = t_max

    def normalize(
            self,
            data: Tensor,
            ts: Optional[Tensor] = None,
            enable_grad: bool = False,
            bound: float = pi / 2,
            rtol: float = 1e-7,
            atol: float = 1e-9,
            verbose: bool = False
    ):
        """ Flow data distribution to base distribution via vector field

        :param data: (batch_dims, 3) tensor of data points on the unit sphere
        :param ts: optional (num_steps,) tensor of times at which to return trajectory values
        :param enable_grad: whether gradients are computed
        :param bound: bound on tangent vector norm to trigger chart switches
        :param rtol: integrator relative tolerance
        :param atol: integrator absolute tolerance
        :param verbose: print integration details
        :return: (batch_dims, 3) tensor of final trajectory values or (num_steps, batch_dims, 3) trajectories
        """

        assert data.shape[-1] == 3
        assert data.dtype == float64

        compute_trajectory = ts is not None
        if compute_trajectory:
            assert ts.dtype == float64
            assert ts[0] == 0.
            assert ts[-1] == self.t_max
            assert ts.diff().min() > 0

        with set_grad_enabled(enable_grad):
            x_curr = data
            t_curr = tensor(0, dtype=float64)
            if compute_trajectory:
                trajectory = zeros(ts.shape + data.shape, dtype=float64)

            num_events = -1
            while t_curr < self.t_max:
                # Compute event time t_next
                num_events += 1
                local_model = SphereVectorFieldTangentRepresentation(self.model, x_curr, t_max=self.t_max, bound=bound)
                v_curr = zeros_like(x_curr)
                t_next, vs = odeint_event(local_model, v_curr, t_curr, event_fn=local_model.event_fn, rtol=rtol, atol=atol)
                v_next = vs[-1]
                x_next = local_model.to_manifold(v_next)

                if compute_trajectory:
                    # Compute solution between t_curr and t_next
                    is_between_events = logical_and(ts >= t_curr, ts <= t_next)
                    ts_between_events = ts[is_between_events]
                    if len(ts_between_events) > 0:
                        include_t_curr = ts_between_events[0] != t_curr
                        if include_t_curr:
                            ts_between_events = cat([t_curr.unsqueeze(-1), ts_between_events])

                        vs_between_events = odeint(local_model, v_curr, ts_between_events, rtol=rtol, atol=atol)
                        xs_between_events = local_model.to_manifold(vs_between_events)
                        if include_t_curr:
                            xs_between_events = xs_between_events[1:]
                        trajectory[is_between_events] = xs_between_events

                if verbose:
                    print(num_events, t_curr.item(), t_next.item())
                t_curr = t_next
                x_curr = x_next

        if compute_trajectory:
            return trajectory

        return x_curr

    def generate(
            self,
            noise: Tensor,
            ts: Optional[Tensor] = None,
            enable_grad: bool = False,
            bound: float = pi / 2,
            rtol: float = 1e-7,
            atol: float = 1e-9,
            verbose: bool = False
    ):
        """ Flow base distribution to data distribution via reversed vector field

        :param noise: (batch_dims, 3) tensor of base distribution samples on the unit sphere
        :param ts: optional (num_steps,) tensor of times at which to return trajectory values
        :param bound: bound on tangent vector norm to trigger chart switches
        :param rtol: integrator relative tolerance
        :param atol: integrator absolute tolerance
        :param enable_grad: whether gradients are computed
        :param verbose: print integration details
        :return: (batch_dims, 3) tensor of final trajectory values or (num_steps, batch_dims, 3) trajectories
        """

        assert noise.shape[-1] == 3
        assert noise.dtype == float64

        compute_trajectory = ts is not None
        if compute_trajectory:
            assert ts.dtype == float64
            assert ts[0] == 0.
            assert ts[-1] == self.t_max
            assert ts.diff().min() > 0

        with set_grad_enabled(enable_grad):
            x_curr = noise
            t_curr = self.t_max
            if compute_trajectory:
                trajectory = zeros(ts.shape + noise.shape, dtype=float64)

            num_events = -1
            while t_curr > 0.:
                # Compute event time t_prev
                num_events += 1
                local_model = SphereVectorFieldTangentRepresentation(self.model, x_curr, t_max=self.t_max, reverse_time=True, bound=bound)
                v_curr = zeros_like(x_curr)
                t_prev, vs = odeint_event(local_model, v_curr, t_curr, event_fn=local_model.event_fn, reverse_time=True, rtol=rtol, atol=atol)
                v_prev = vs[-1]
                x_prev = local_model.to_manifold(v_prev)

                if compute_trajectory:
                    # Compute solution between t_prev and t_curr
                    is_between_events = logical_and(ts >= t_prev, ts <= t_curr)
                    ts_between_events = ts[is_between_events]
                    if len(ts_between_events) > 0:
                        include_t_curr = ts_between_events[-1] != t_curr
                        if include_t_curr:
                            ts_between_events = cat([ts_between_events, t_curr.unsqueeze(-1)])

                        vs_between_events = odeint(local_model, v_curr, ts_between_events.flip(0,), rtol=rtol, atol=atol).flip(0,)
                        xs_between_events = local_model.to_manifold(vs_between_events)
                        if include_t_curr:
                            xs_between_events = xs_between_events[:-1]
                        trajectory[is_between_events] = xs_between_events

                if verbose:
                    print(num_events, t_curr.item(), t_prev.item())
                t_curr = t_prev
                x_curr = x_prev

        if compute_trajectory:
            return trajectory

        return x_curr

    def augmented_generate(
            self,
            noise: Tensor,
            log_prob: Tensor,
            ts: Optional[Tensor] = None,
            enable_grad: bool = False,
            bound: float = pi / 2,
            rtol: float = 1e-7,
            atol: float = 1e-9,
            verbose: bool = False
    ):
        """ Flow base distribution (with log probabilities) to data distribution via reversed vector field

        :param noise: (batch_dims, 3) tensor of base distribution samples on the unit sphere
        :param log_prob: (batch_dims,) tensor of log probabilities under base distribution
        :param ts: optional (num_steps,) tensor of times at which to return trajectory values
        :param enable_grad: whether gradients are computed
        :param bound: bound on tangent vector norm to trigger chart switches
        :param rtol: integrator relative tolerance
        :param atol: integrator absolute tolerance
        :param verbose: print integration details
        :return: (batch_dims, 3) tensor of final trajectory values and (batch_dims,) tensor of final log probabilities
                    or (num_steps, batch_dims, 3) trajectories and (num_steps, batch_dims) log probability trajectories
        """

        assert noise.shape[-1] == 3
        assert noise.dtype == float64
        assert log_prob.shape == noise.shape[:-1]
        assert log_prob.dtype == float64

        compute_trajectory = ts is not None
        if compute_trajectory:
            assert ts.dtype == float64
            assert ts[0] == 0.
            assert ts[-1] == self.t_max
            assert ts.diff().min() > 0

        with set_grad_enabled(enable_grad):
            x_curr = noise
            log_prob_curr = log_prob
            aug_shape = tensor(x_curr.shape)
            aug_shape[-1] += 1
            aug_shape = Size(aug_shape)
            t_curr = self.t_max
            if compute_trajectory:
                trajectory = zeros(ts.shape + aug_shape, dtype=float64)

            num_events = -1
            while t_curr > 0.:
                # Compute event time t_prev
                num_events += 1
                local_model = SphereVectorFieldTangentRepresentation(self.model, x_curr, t_max=self.t_max, reverse_time=True, bound=bound)
                aug_local_model = AugmentedSphereVectorFieldTangentRepresentation(local_model)
                v_aug_curr = zeros(aug_shape, dtype=float64)
                v_aug_curr[..., -1] = log_prob_curr
                t_prev, v_augs = odeint_event(aug_local_model, v_aug_curr, t_curr, event_fn=aug_local_model.event_fn, reverse_time=True, rtol=rtol, atol=atol)
                v_aug_prev = v_augs[-1]
                v_prev = v_aug_prev[..., :-1]
                x_prev = local_model.to_manifold(v_prev)
                log_prob_prev = v_aug_prev[..., -1]

                if compute_trajectory:
                    # Compute solution between t_prev and t_curr
                    is_between_events = logical_and(ts >= t_prev, ts <= t_curr)
                    ts_between_events = ts[is_between_events]
                    if len(ts_between_events) > 0:
                        include_t_curr = ts_between_events[-1] != t_curr
                        if include_t_curr:
                            ts_between_events = cat([ts_between_events, t_curr.unsqueeze(-1)])

                        v_augs_between_events = odeint(aug_local_model, v_aug_curr, ts_between_events.flip(0,), rtol=rtol, atol=atol).flip(0,)
                        vs_between_events = v_augs_between_events[..., :-1]
                        xs_between_events = local_model.to_manifold(vs_between_events)
                        log_probs_between_events = v_augs_between_events[..., -1]
                        zs_between_events = cat([xs_between_events, log_probs_between_events.unsqueeze(-1)], dim=-1)
                        if include_t_curr:
                            zs_between_events = zs_between_events[:-1]
                        trajectory[is_between_events] = zs_between_events

                if verbose:
                    print(num_events, t_curr.item(), t_prev.item())
                t_curr = t_prev
                x_curr = x_prev
                log_prob_curr = log_prob_prev

        if compute_trajectory:
            return trajectory[..., :-1], trajectory[..., -1]

        return x_curr, log_prob_curr

    def log_prob(
            self,
            data: Tensor,
            enable_grad: bool = True,
            bound: float = pi / 2,
            rtol: float = 1e-7,
            atol: float = 1e-9,
            verbose: bool = False
    ) -> Tensor:
        """ Compute log probabilities of data

        :param data: (batch_dims, 3) tensor of data on the unit sphere
        :param enable_grad: whether gradients are computed
        :param bound: bound on tangent vector norm to trigger chart switches
        :param rtol: integrator relative tolerance
        :param atol: integrator absolute tolerance
        :param verbose: print integration details
        :return: (batch_dims,) tensor of log probabilities
        """
        assert data.shape[-1] == 3
        assert (norm(data, dim=-1) - 1.).abs().max() < 1e-12

        noise = self.normalize(data, enable_grad=enable_grad, rtol=rtol, atol=atol, verbose=verbose, bound=bound)
        noise_log_prob = self.base_distribution.log_prob(noise)
        reconstructed_data, data_log_prob = self.augmented_generate(noise, noise_log_prob, enable_grad=enable_grad, rtol=rtol, atol=atol, verbose=verbose, bound=bound)

        with no_grad():
            assert norm(reconstructed_data - data, dim=-1).abs().max() < 1e-6

        return data_log_prob
