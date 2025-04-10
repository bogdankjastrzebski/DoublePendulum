from double_pendulum import *
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np  # For initial state and time array creation.


def observe(state, dist, project=project):
    return project(state) + dist.sample()


def make_simulation(
            state, dt, step_size, steps, M1, M2, L1, L2, G, dx=0.01,
            dist=torch.distributions.MultivariateNormal(
                torch.zeros(2), 0.01 * torch.eye(2)
            ),
        ):
    return [state] + [
        state := simulate(state, dt, step_size, M1, M2, L1, L2, G)
        for _ in range(steps)
    ]


def test_gradient_finite_difference_agreement(
        initial_state, dt, n, M1, M2, L1, L2, G, dx=0.01):

    """Tests that auto-diff gradients align with finite diff."""
    def e(i, n=4):
        r = torch.zeros(n)
        r[i] = 1.0
        return r

    y_h0 = simulate(initial_state + dt * e(0), dt, n, M1, M2, L1, L2, G)
    y_h1 = simulate(initial_state + dt * e(1), dt, n, M1, M2, L1, L2, G)
    y_h2 = simulate(initial_state + dt * e(2), dt, n, M1, M2, L1, L2, G)
    y_h3 = simulate(initial_state + dt * e(3), dt, n, M1, M2, L1, L2, G)
    y_0 = simulate(initial_state, dt, n, M1, M2, L1, L2, G)

    d0 = (y_h0 - y_0)/dx
    d1 = (y_h1 - y_0)/dx
    d2 = (y_h2 - y_0)/dx
    d3 = (y_h3 - y_0)/dx

    d = torch.stack([d0, d1, d2, d3])

    def compute_grad(func, initial_state, dt, n, M1, M2, L1, L2, G):
        """Compute the Jacobian of the final state
        with respect to the initial state."""
        initial_state.requires_grad_(True)
        final_state = func(simulate(initial_state, dt, n, M1, M2, L1, L2, G))
        return torch.autograd.grad(
            final_state,
            initial_state,
            create_graph=False,
            retain_graph=False,
        )[0]

    def grad_lambda(i):
        return compute_grad(
            lambda x: x[i], initial_state, dt, n, M1, M2, L1, L2, G
        )

    grad0 = grad_lambda(0) 
    grad1 = grad_lambda(1) 
    grad2 = grad_lambda(2) 
    grad3 = grad_lambda(3) 

    grad = torch.stack([grad0, grad1, grad2, grad3]).T
    # now d and grad should be similar, not allclose, but similar.
    # Assert some kind of agreement here
    print('Finite Difference: ', d)
    print('Autodiff: ', grad)


def test_log_likelihood_evaluates(
            initial_state, dt, n, M1, M2, L1, L2, G,
            dist=torch.distributions.MultivariateNormal(
                torch.zeros(2), 0.01 * torch.eye(2)
            ),
        ):
    """Tests that log_likelihood evaluates without errors."""
    ts = [0, n]
    y_0 = simulate(initial_state, dt, n, M1, M2, L1, L2, G)
    xs = [
        project(initial_state).detach() + dist.sample(),
        project(y_0).detach() + dist.sample(),
    ]
    print("Log Likelihood:", log_likelihood(ts, xs, initial_state, dt, M1, M2, L1, L2, G, dist))


def test_optimization_improves_likelihood(
            initial_state, dt, step_size, steps, M1, M2, L1, L2, G,
            dist=torch.distributions.MultivariateNormal(
                torch.zeros(2), 0.01 * torch.eye(2)
            ),
        ):
    """Tests that optimization increases log-likelihood."""
    initial_state = initial_state.detach()

    ts = torch.arange(steps + 1) * step_size # [0, n]
    hs = make_simulation(
        initial_state, dt, step_size, steps,
        M1, M2, L1, L2, G,
    )
    print('hs', len(hs), hs[-1].shape, hs[-1])
    xs = list(map(lambda x: observe(x, dist), hs))
    print('xs', len(xs), xs[-1].shape, xs[-1])

    
    estimated_state = estimate_parameters(
        ts, xs, dt,
        M1, M2, L1, L2, G,
        dist,
    )
    estimated_hs = make_simulation(
        estimated_state, dt, step_size, steps,
        M1, M2, L1, L2, G
    )
    
    optimized_likelihood = log_likelihood(
        ts, xs, estimated_state,
        dt, M1, M2, L1, L2, G,
        dist,
    )
    initial_likelihood = log_likelihood(
        ts, xs, initial_state,
        dt, M1, M2, L1, L2, G,
        dist,
    )

    print("=== Optimization Results ===")
    print("Initial State:", initial_state)
    print("Estimated State:", estimated_state)

    print("---")
    print("Initial Likelihood:", initial_likelihood.item())
    print("Optimized Likelihood:", optimized_likelihood.item())

    print("---")
    print("Simulated last state from Initial State:", )
    print("Simulated last state from Estimated State:", )
    print("---")

    print("Difference (Y_estimated - Y_0):", y_estimated - y_0)
    print("Difference (Estimated State - Initial State):",
          estimated_state - initial_state)

    assert optimized_likelihood > initial_likelihood, (
        "Optimization failed to improve log-likelihood. "
        f"Initial: {initial_likelihood}, Optimized:"
        f" {optimized_likelihood}."
    )


def animate(traj, L1, L2):
    # Constants (can be moved to a config file)
    L = 2.0  # max combined length for animation
    """Animates the double pendulum trajectory."""
    x1 =  L1 * traj[:, 0].sin()
    y1 = -L1 * traj[:, 0].cos()
    x2 =  L2 * traj[:, 2].sin() + x1
    y2 = -L2 * traj[:, 2].cos() + y1

    fig = plt.figure(figsize=(5,4))
    plt.plot(traj[:, 0])
    plt.plot(traj[:, 2])
    plt.show()

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    trace, = ax.plot([], [], '.-', lw=1, ms=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def animation_frame(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]
        history_x = x2[:i]
        history_y = y2[:i]
        line.set_data(thisx, thisy)
        trace.set_data(history_x, history_y)
        time_text.set_text(time_template % (i * dt.numpy()))
        return line, trace, time_text

    ani = animation.FuncAnimation(
        fig, animation_frame, len(traj), interval=dt.numpy() * 1000,
        blit=True,
    )

    plt.show()


if __name__ == '__main__':
    M1 = torch.tensor(1.0)
    M2 = torch.tensor(1.0)
    L1 = torch.tensor(1.0)
    L2 = torch.tensor(1.0)
    G = torch.tensor(9.8)
    dt = torch.tensor(0.01)
    step_size = 100
    steps = 1
    initial_state = torch.tensor([
        np.pi/2, 0.0, np.pi / 2, 0.0
    ], dtype=torch.float64)

    # traj = trajectory(
    #     initial_state, dt, step_size,
    #     M1, M2, L1, L2, G
    # ).detach()
    # animate(traj, L1, L2)

    test_gradient_finite_difference_agreement(
        initial_state, dt, step_size,
        M1, M2, L1, L2, G,
    )
    test_log_likelihood_evaluates(
        initial_state, dt, step_size,
        M1, M2, L1, L2, G,
    )
    test_optimization_improves_likelihood(
        initial_state, dt, step_size, steps,
        M1, M2, L1, L2, G,
    )
