import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np  # For initial state and time array creation.
from types import FunctionType
# from torch.autograd.functional import jacobian
from torch.func import jacrev


# def derivs(state, M1, M2, L1, L2, G):
#     """Computes the derivatives for the double pendulum."""
#     theta1 = state[0]
#     omega1 = state[1]
#     theta2 = state[2]
#     omega2 = state[3]
# 
#     dtheta1_dt = omega1
# 
#     delta = theta2 - theta1
#     cos_delta = torch.cos(delta)
#     sin_delta = torch.sin(delta)
# 
#     den1 = (M1 + M2) * L1 - M2 * L1 * cos_delta * cos_delta
#     domega1_dt = (
#         M2 * L1 * omega1 * omega1 * sin_delta * cos_delta
#         + M2 * G * torch.sin(theta2) * cos_delta
#         + M2 * L2 * omega2 * omega2 * sin_delta
#         - (M1 + M2) * G * torch.sin(theta1)
#     ) / den1
# 
#     dtheta2_dt = omega2
# 
#     den2 = (L2 / L1) * den1
#     domega2_dt = (
#         - M2 * L2 * omega2 * omega2 * sin_delta * cos_delta
#         + (M1 + M2) * G * torch.sin(theta1) * cos_delta
#         - (M1 + M2) * L1 * omega1 * omega1 * sin_delta
#         - (M1 + M2) * G * torch.sin(theta2)
#     ) / den2
# 
#     dydx = torch.stack([
#         dtheta1_dt,
#         domega1_dt,
#         dtheta2_dt,
#         domega2_dt,
#     ])
#     return dydx

def double_pendulum_derivatives(
            state,
            mass_1,
            mass_2,
            length_1,
            length_2,
            gravity,
        ):
    """Computes the derivatives of the state variables for a double pendulum.

    Args:
        state (torch.Tensor): A tensor of shape (4,) containing the current state
            of the pendulum: [theta_1, omega_1, theta_2, omega_2], where
            theta_1 is the angle of the first pendulum from the vertical,
            omega_1 is the angular velocity of the first pendulum,
            theta_2 is the angle of the second pendulum from the vertical, and
            omega_2 is the angular velocity of the second pendulum.
        mass_1 (float): The mass of the first pendulum bob.
        mass_2 (float): The mass of the second pendulum bob.
        length_1 (float): The length of the first pendulum arm.
        length_2 (float): The length of the second pendulum arm.
        gravity (float): The acceleration due to gravity.

    Returns:
        torch.Tensor: A tensor of shape (4,) containing the derivatives of the
            state variables: [dtheta_1/dt, domega_1/dt, dtheta_2/dt, domega_2/dt].
    """
    theta_1 = state[0]
    omega_1 = state[1]
    theta_2 = state[2]
    omega_2 = state[3]

    dtheta_1_dt = omega_1

    delta = theta_2 - theta_1
    cos_delta = torch.cos(delta)
    sin_delta = torch.sin(delta)

    den1 = (mass_1 + mass_2) * length_1 - mass_2 * length_1 * cos_delta * cos_delta

    domega_1_dt = (
        mass_2 * length_1 * omega_1 * omega_1 * sin_delta * cos_delta
        + mass_2 * gravity * torch.sin(theta_2) * cos_delta
        + mass_2 * length_2 * omega_2 * omega_2 * sin_delta
        - (mass_1 + mass_2) * gravity * torch.sin(theta_1)
    ) / den1

    dtheta_2_dt = omega_2

    den2 = (length_2 / length_1) * den1
    domega_2_dt = (
        -mass_2 * length_2 * omega_2 * omega_2 * sin_delta * cos_delta
        + (mass_1 + mass_2) * gravity * torch.sin(theta_1) * cos_delta
        - (mass_1 + mass_2) * length_1 * omega_1 * omega_1 * sin_delta
        - (mass_1 + mass_2) * gravity * torch.sin(theta_2)
    ) / den2

    dstate_dt = torch.stack([
        dtheta_1_dt,
        domega_1_dt,
        dtheta_2_dt,
        domega_2_dt,
    ])
    return dstate_dt


def update(y, dt, m1, m2, l1, l2, g):
    """Updates the state using a single Euler step."""
    return y + double_pendulum_derivatives(y, m1, m2, l1, l2, g) * dt


def simulate(y, dt, n, m1, m2, l1, l2, g):
    """Simulates the double pendulum for n steps."""
    for _ in range(n):
        y = update(y, dt, m1, m2, l1, l2, g)
    return y


def trajectory(y, dt, n, m1, m2, l1, l2, g):
    """Calculates the trajectory of the double pendulum."""
    return torch.stack([
        y := update(y, dt, m1, m2, l1, l2, g)
        for _ in range(n)
    ])


def project(h):
    """Projects the state vector to position (theta1, theta2)."""
    return h[[0, 2]]


def log_likelihood(ts, xs, h, dt, m1, m2, l1, l2, g, dist, project=project):
    """Calculates the log-likelihood of the measurements."""
    s = dist.log_prob(xs[0] - project(h)) / len(ts)
    for i in range(len(ts) - 1):
        steps = ts[i+1] - ts[i]
        h = simulate(h, dt, steps, m1, m2, l1, l2, g)
        s += dist.log_prob(xs[i+1] - project(h)) / len(ts)
    return s


def positions(ts, h, dt, m1, m2, l1, l2, g, project=project):
    """Calculates the positions, that can be used to calculate log-likelihood."""
    states = [project(h)]
    for i in range(len(ts) - 1):
        steps = ts[i+1] - ts[i]
        h = simulate(h, dt, steps, m1, m2, l1, l2, g)
        states.append(project(h))
    return torch.stack(states)


def log_likelihood_from_positions(ps, xs, dist):
    """Calculates log-likelihood from positions."""
    return dist.log_prob(xs - ps) / ps.shape[0]


INITIAL = lambda: 0.10 * torch.ones(4)
#lambda: 0.1 * torch.randn(4)

def estimate_parameters_sgd(
            ts, xs, dt, M1, M2, L1, L2, G,
            dist=torch.distributions.MultivariateNormal(
                torch.zeros(2), 0.1 * torch.eye(2)
            ),
            project=project,
            iter=50,
            initial=INITIAL,
        ):
    """Estimates initial state parameters by maximizing log-likelihood."""
    losses = []
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
        losses.append(loss.item())
    return hidden, losses


def estimate_parameters_lbfgs(
            ts, xs, dt, M1, M2, L1, L2, G,
            dist=torch.distributions.MultivariateNormal(
                torch.zeros(2), 0.1 * torch.eye(2)
            ),
            project=project,
            iter=10,
            lr=0.05,
            max_iter=10,
            history_size=30,
            initial=INITIAL,
        ):
    """Estimates initial state parameters by maximizing log-likelihood."""
    losses = []
    if type(initial) is FunctionType:
        initial = initial()
        initial[[0, 2]] = xs[0].float()
    hidden = torch.nn.Parameter(initial)
    opt = torch.optim.LBFGS([hidden], lr=lr, max_iter=max_iter, history_size=4)
    for _ in (pbar := tqdm(range(iter))):
        def closure():
            opt.zero_grad()
            loss = -log_likelihood(
                ts, xs, hidden, dt, M1, M2, L1, L2, G, dist, project
            )
            loss.backward()
            # hidden.grad.data.clamp_(-100, 100) 
            pbar.set_description(f"{loss.item()}")
            losses.append(loss.item())
            return loss
        opt.step(closure)
    return hidden, losses


def estimate_parameters_gn(
            ts, xs, dt, m1, m2, l1, l2, g,
            dist=torch.distributions.MultivariateNormal(
                torch.zeros(2), 0.1 * torch.eye(2)
            ),
            project=project,
            iter=20,
            lr=100,
            initial=INITIAL,
        ):
    """Estimates initial state parameters by maximizing log-likelihood."""
    losses = []
    if type(initial) is FunctionType:
        initial = initial()
        initial[[0, 2]] = xs[0].float()
    hidden = initial
    xs = torch.stack(xs).float()
    def halfmodel(h):
        ps = positions(ts, h, dt, m1, m2, l1, l2, g, project)
        return ps, ps
    for _ in (pbar := tqdm(range(iter))):

        jac, ps = jacrev(halfmodel, has_aux=True)(hidden)
        jac = jac.flatten(0, 1)
        hes = jac.T @ jac + torch.eye(jac.shape[1]) / lr

        # print("Xs: ", xs.shape, xs.dtype)
        # print("Ps: ", ps.shape, ps.dtype)
        # print("", (ps - xs).flatten().shape)
        # print("", (jac.T @ (ps - xs).flatten()).shape)

        hidden = hidden - torch.linalg.solve(hes, jac.T @ (ps - xs).flatten())

        # print("Xs: ", xs.shape)
        # print("Jacobian: ", jac.shape)
        # print("Hes: ", hes.shape, hes)
        # # print("Positions: ", halfmodel(hidden))
        # print("Positions: ", ps)
        # print("Hidden: ", hidden.shape)
        # print("Hidden: ", hidden)

        with torch.no_grad():
            L = - log_likelihood(
                ts, xs, hidden, dt, m1, m2, l1, l2, g, dist, project=project
            )
                # raise Exception("Hello darkness my old friend...")
            # L = log_likelihood_from_positions(ps, xs, dist)
            pbar.set_description(f"{L.item()}")
            losses.append(L.item())
    return hidden, losses


