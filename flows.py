"""Flow-matching pieces shared by every version: the training loss, the ODE samplers and EMA.

Time convention (used everywhere in this repo):
    t = 0  ->  source / noise   x_0 ~ N(0, I)
    t = 1  ->  data             x_1 ~ p_data
The straight ("rectified" / linear) path between a noise sample and a data sample is
    x_t = (1 - t) x_0 + t x_1,          so  dx_t/dt = x_1 - x_0.
Training regresses v_theta(x_t, t) onto x_1 - x_0; sampling integrates dx/dt = v_theta(x, t)
from t = 0 (noise) to t = 1 (data).
"""
import copy

import torch


# ---------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------
def sample_path(x0, x1, t=None):
    """A random time t ~ U(0, 1) per example, the point x_t on the straight line from x0
    (noise) to x1 (data), and the regression target u_t = x1 - x0 (the line's velocity)."""
    if t is None:
        t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)
    t_ = t.view(-1, *([1] * (x0.dim() - 1)))
    x_t = (1 - t_) * x0 + t_ * x1
    return t, x_t, x1 - x0


def cfm_loss(model, x0, x1, cond=None):
    """Conditional flow matching loss  E || v(x_t, t) - (x1 - x0) ||^2."""
    t, x_t, u_t = sample_path(x0, x1)
    v = model(x_t, t) if cond is None else model(x_t, t, cond)
    return ((v - u_t) ** 2).mean()


# ---------------------------------------------------------------------------------------
# Sampling: integrate dx/dt = v(x, t) from t = 0 to t = 1
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def integrate(velocity, x0, steps=50, method='euler', return_path=False):
    """Fixed-step ODE solvers. `velocity(x, t)` takes a batch x and a float t in [0, 1].

    method    evaluations of v per step   global error
    euler     1                           O(h)
    midpoint  2                           O(h^2)
    heun      2                           O(h^2)     (trapezoidal predictor-corrector)
    rk4       4                           O(h^4)
    Compare solvers at equal *network evaluations* (NFE = steps x evaluations per step).
    """
    x = x0
    path = [x0]
    ts = torch.linspace(0, 1, steps + 1).tolist()
    for t, t_next in zip(ts[:-1], ts[1:]):
        h = t_next - t
        if method == 'euler':
            x = x + h * velocity(x, t)
        elif method == 'midpoint':
            k1 = velocity(x, t)
            x = x + h * velocity(x + 0.5 * h * k1, t + 0.5 * h)
        elif method == 'heun':
            k1 = velocity(x, t)
            k2 = velocity(x + h * k1, t_next)
            x = x + 0.5 * h * (k1 + k2)
        elif method == 'rk4':
            k1 = velocity(x, t)
            k2 = velocity(x + 0.5 * h * k1, t + 0.5 * h)
            k3 = velocity(x + 0.5 * h * k2, t + 0.5 * h)
            k4 = velocity(x + h * k3, t_next)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(method)
        if return_path:
            path.append(x)
    return torch.stack(path) if return_path else x


NFE_PER_STEP = {'euler': 1, 'midpoint': 2, 'heun': 2, 'rk4': 4}


def as_velocity(model, cond=None):
    """Wrap a network so that integrate() can call it with a python float t."""
    def v(x, t):
        tt = torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
        return model(x, tt) if cond is None else model(x, tt, cond)
    return v


# ---------------------------------------------------------------------------------------
# Exponential moving average of the weights
# ---------------------------------------------------------------------------------------
class EMA:
    """Keeps theta_ema <- decay * theta_ema + (1 - decay) * theta after every step.

    The raw weights jitter around because every step sees a different noisy estimate of
    the loss; the average is a much smoother velocity field and gives visibly cleaner
    samples. The decay warms up as (1 + n) / (10 + n) so early averages are not
    dominated by the random initialisation."""
    def __init__(self, model, decay=0.9995):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for p_ema, p in zip(self.model.parameters(), model.parameters()):
            p_ema.lerp_(p.detach(), 1 - d)
        for b_ema, b in zip(self.model.buffers(), model.buffers()):
            b_ema.copy_(b)
