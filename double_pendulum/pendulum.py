import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np  # For initial state and time array creation.
from types import FunctionType


def derivs(state, M1, M2, L1, L2, G):
    """Computes the derivatives for the double pendulum."""
    theta1 = state[0]
    omega1 = state[1]
    theta2 = state[2]
    omega2 = state[3]

    dtheta1_dt = omega1

    delta = theta2 - theta1
    cos_delta = torch.cos(delta)
    sin_delta = torch.sin(delta)

    den1 = (M1 + M2) * L1 - M2 * L1 * cos_delta * cos_delta
    domega1_dt = (
        M2 * L1 * omega1 * omega1 * sin_delta * cos_delta
        + M2 * G * torch.sin(theta2) * cos_delta
        + M2 * L2 * omega2 * omega2 * sin_delta
        - (M1 + M2) * G * torch.sin(theta1)
    ) / den1

    dtheta2_dt = omega2

    den2 = (L2 / L1) * den1
    domega2_dt = (
        - M2 * L2 * omega2 * omega2 * sin_delta * cos_delta
        + (M1 + M2) * G * torch.sin(theta1) * cos_delta
        - (M1 + M2) * L1 * omega1 * omega1 * sin_delta
        - (M1 + M2) * G * torch.sin(theta2)
    ) / den2

    dydx = torch.stack([
        dtheta1_dt,
        domega1_dt,
        dtheta2_dt,
        domega2_dt,
    ])
    return dydx

def update(y, dt, M1, M2, L1, L2, G):
    """Updates the state using a single Euler step."""
    return y + derivs(y, M1, M2, L1, L2, G) * dt

def simulate(y, dt, n, M1, M2, L1, L2, G):
    """Simulates the double pendulum for n steps."""
    for _ in range(n):
        y = update(y, dt, M1, M2, L1, L2, G)
    return y


def trajectory(y, dt, n, M1, M2, L1, L2, G):
    """Calculates the trajectory of the double pendulum."""
    return torch.stack([
        y := update(y, dt, M1, M2, L1, L2, G)
        for _ in range(n)
    ])


def project(h):
    """Projects the state vector to position (theta1, theta2)."""
    return h[[0, 2]]


def log_likelihood(ts, xs, h, dt, M1, M2, L1, L2, G, dist, project=project):
    """Calculates the log-likelihood of the measurements."""
    s = dist.log_prob(xs[0] - project(h)) / len(ts)
    for i in range(len(ts) - 1):
        steps = ts[i+1] - ts[i]
        h = simulate(h, dt, steps, M1, M2, L1, L2, G)
        s += dist.log_prob(xs[i+1] - project(h)) / len(ts)
    return s


def estimate_parameters(
            ts, xs, dt, M1, M2, L1, L2, G,
            dist=torch.distributions.MultivariateNormal(
                torch.zeros(2), 0.1 * torch.eye(2)
            ),
            project=project,
            iter=50,
            initial=lambda: 0.1 * torch.randn(4),
        ):
    """Estimates initial state parameters by maximizing log-likelihood."""
    if type(initial) is FunctionType:
        initial = initial()
        initial[[0,2]] = xs[0].float()
    hidden = torch.nn.Parameter(initial)
    opt = torch.optim.SGD([hidden], lr=0.01, momentum=0.0)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda i: 0.1/(i + 1))
    for _ in (pbar := tqdm(range(iter))):
        opt.zero_grad()
        loss = -log_likelihood(
            ts, xs, hidden, dt, M1, M2, L1, L2, G, dist, project
        )
        loss.backward()
        hidden.grad.data.clamp_(-100.0, 100.0) 
        opt.step()
        sch.step()
        pbar.set_description(f"{loss.item()}")
    return hidden


def estimate_parameters_1(
            ts, xs, dt, M1, M2, L1, L2, G,
            dist=torch.distributions.MultivariateNormal(
                torch.zeros(2), 0.1 * torch.eye(2)
            ),
            project=project,
            iter=10,
            initial=lambda: 0.1 * torch.randn(4),
        ):
    """Estimates initial state parameters by maximizing log-likelihood."""
    if type(initial) is FunctionType:
        initial = initial()
        initial[[0, 2]] = xs[0].float()
    hidden = torch.nn.Parameter(initial)
    opt = torch.optim.LBFGS([hidden], lr=0.05, max_iter=10, history_size=4)
    for _ in (pbar := tqdm(range(iter))):
        def closure():
            opt.zero_grad()
            loss = -log_likelihood(
                ts, xs, hidden, dt, M1, M2, L1, L2, G, dist, project
            )
            loss.backward()
            # hidden.grad.data.clamp_(-100.0, 100.0) 
            pbar.set_description(f"{loss.item()}")
            return loss
        opt.step(closure)
    return hidden
