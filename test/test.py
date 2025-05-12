from double_pendulum import *
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np  # For initial state and time array creation.
import time
import os

torch.manual_seed(0)

def observe(state, dist, project=project):
    return project(state) + dist.sample()


def make_simulation(
            state, dt, step_size, steps, m1, m2, l1, l2, g, dx=0.01,
            dist=torch.distributions.MultivariateNormal(
                torch.zeros(2), 0.01 * torch.eye(2)
            ),
        ):
    return [state] + [
        state := simulate(state, dt, step_size, m1, m2, l1, l2, g)
        for _ in range(steps)
    ]


def test_gradient_finite_difference_agreement(
        initial_state, dt, n, m1, m2, l1, l2, g, dx=0.01):

    """Tests that auto-diff gradients align with finite diff."""
    def e(i, n=4):
        r = torch.zeros(n)
        r[i] = 1.0
        return r

    y_h0 = simulate(initial_state + dt * e(0), dt, n, m1, m2, l1, l2, g)
    y_h1 = simulate(initial_state + dt * e(1), dt, n, m1, m2, l1, l2, g)
    y_h2 = simulate(initial_state + dt * e(2), dt, n, m1, m2, l1, l2, g)
    y_h3 = simulate(initial_state + dt * e(3), dt, n, m1, m2, l1, l2, g)
    y_0 = simulate(initial_state, dt, n, m1, m2, l1, l2, g)

    d0 = (y_h0 - y_0)/dx
    d1 = (y_h1 - y_0)/dx
    d2 = (y_h2 - y_0)/dx
    d3 = (y_h3 - y_0)/dx

    d = torch.stack([d0, d1, d2, d3])

    def compute_grad(func, initial_state, dt, n, m1, m2, l1, l2, g):
        """Compute the Jacobian of the final state
        with respect to the initial state."""
        initial_state.requires_grad_(True)
        final_state = func(simulate(initial_state, dt, n, m1, m2, l1, l2, g))
        return torch.autograd.grad(
            final_state,
            initial_state,
            create_graph=False,
            retain_graph=False,
        )[0]

    def grad_lambda(i):
        return compute_grad(
            lambda x: x[i], initial_state, dt, n, m1, m2, l1, l2, g
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
            initial_state, dt, n, m1, m2, l1, l2, g,
            dist=torch.distributions.MultivariateNormal(
                torch.zeros(2), 0.01 * torch.eye(2)
            ),
        ):
    """Tests that log_likelihood evaluates without errors."""
    ts = [0, n]
    y_0 = simulate(initial_state, dt, n, m1, m2, l1, l2, g)
    xs = [
        project(initial_state).detach() + dist.sample(),
        project(y_0).detach() + dist.sample(),
    ]
    print("Log Likelihood:", log_likelihood(
        ts, xs, initial_state, dt, m1, m2, l1, l2, g, dist))


def test_optimization_improves_likelihood(
            initial_state, dt, step_size, steps, m1, m2, l1, l2, g,
            dist=torch.distributions.MultivariateNormal(
                torch.zeros(2), 0.01 * torch.eye(2)
            ),
        ):
    """Tests that optimization increases log-likelihood."""
    if os.path.isfile('tmp/results.pt'):
        results = torch.load('tmp/results.pt')
        (ts, xs, hs,
            t0_gn, t1_gn,
            t0_sgd, t1_sgd,
            t0_lbfgs, t1_lbfgs,
            estimated_state_gn,
            estimated_state_sgd,
            estimated_state_lbfgs,
            losses_gn,
            losses_sgd,
            losses_lbfgs,
            estimated_hs_gn,
            estimated_hs_sgd,
            estimated_hs_lbfgs,
            ) = results
    else:
        initial_state = initial_state.detach()

        ts = torch.arange(steps + 1) * step_size # [0, n]
        hs = make_simulation(
            initial_state, dt, step_size, steps,
            m1, m2, l1, l2, g,
        )
        print('hs', len(hs), hs[-1].shape, hs[-1])
        xs = list(map(lambda x: observe(x, dist), hs))
        print('xs', len(xs), xs[-1].shape, xs[-1])
        
        
        t0_gn = time.time()
        estimated_state_gn, losses_gn = estimate_parameters_gn(
            ts, xs, dt,
            m1, m2, l1, l2, g,
            dist,
        )
        t1_gn = time.time()
        
        t0_sgd = time.time()
        estimated_state_sgd, losses_sgd = estimate_parameters_sgd(
            ts, xs, dt,
            m1, m2, l1, l2, g,
            dist,
        )
        t1_sgd = time.time()

        t0_lbfgs = time.time()
        estimated_state_lbfgs, losses_lbfgs = estimate_parameters_lbfgs(
            ts, xs, dt,
            m1, m2, l1, l2, g,
            dist,
        )
        t1_lbfgs = time.time()

        estimated_hs_gn = make_simulation(
            estimated_state_gn, dt, step_size, steps,
            m1, m2, l1, l2, g
        )

        estimated_hs_sgd = make_simulation(
            estimated_state_sgd, dt, step_size, steps,
            m1, m2, l1, l2, g
        )

        estimated_hs_lbfgs = make_simulation(
            estimated_state_lbfgs, dt, step_size, steps,
            m1, m2, l1, l2, g
        )
    
        # optimized_likelihood = log_likelihood(
        #     ts, xs, estimated_state,
        #     dt, m1, m2, l1, l2, g,
        #     dist,
        # )

        # initial_likelihood = log_likelihood(
        #     ts, xs, initial_state,
        #     dt, m1, m2, l1, l2, g,
        #     dist,
        # )

        results = (
            ts, xs, hs,
            t0_gn, t1_gn,
            t0_sgd, t1_sgd,
            t0_lbfgs, t1_lbfgs,
            estimated_state_gn,
            estimated_state_sgd,
            estimated_state_lbfgs,
            losses_gn,
            losses_sgd,
            losses_lbfgs,
            estimated_hs_gn,
            estimated_hs_sgd,
            estimated_hs_lbfgs,
        )

        torch.save(results, 'tmp/results.pt')

    # print("=== Optimization Results ===")
    # print("Initial State:", list(initial_state.detach().numpy()))
    # print("Estimated State:", list(estimated_state.detach().numpy()))

    # print("---")
    # print("Initial Likelihood:", initial_likelihood.item())
    # print("Optimized Likelihood:", optimized_likelihood.item())

    print("---")
    # losses_lbfgs = [-initial_likelihood] + losses_lbfgs
    # losses_sgd = [-initial_likelihood] + losses_sgd
    # losses_gn = [-initial_likelihood] + losses_gn
    fig = plt.figure(figsize=(5, 4))
    rng_lbfgs = torch.linspace(0, t1_lbfgs - t0_lbfgs, len(losses_lbfgs))
    rng_sgd   = torch.linspace(0, t1_sgd   - t0_sgd,   len(losses_sgd))
    rng_gn    = torch.linspace(0, t1_gn    - t0_gn,    len(losses_gn))
    plt.plot(rng_lbfgs, losses_lbfgs, label='L-BFGS')
    plt.plot(rng_sgd,   losses_sgd,   label='SGD')
    plt.plot(rng_gn,    losses_gn,    label='Gauss-Newton')
    plt.legend()
    plt.xlabel('time (s)')
    plt.ylabel('loss value')
    plt.title('Convergence of Algorithms')
    plt.savefig('img/curves.pdf')
    plt.show()

    print("---")
    b = 2.4
    fig = plt.figure(figsize=(5, 4))
    rng_lbfgs = torch.linspace(0, t1_lbfgs - t0_lbfgs, len(losses_lbfgs))
    rng_sgd   = torch.linspace(0, t1_sgd   - t0_sgd,   len(losses_sgd))
    rng_gn    = torch.linspace(0, t1_gn    - t0_gn,    len(losses_gn))
    plt.plot(rng_lbfgs, (torch.tensor(losses_lbfgs) + b).log(), label='L-BFGS')
    plt.plot(rng_sgd,   (torch.tensor(losses_sgd) + b).log(),   label='SGD')
    plt.plot(rng_gn,    (torch.tensor(losses_gn) + b).log(),    label='Gauss-Newton')
    plt.legend()
    plt.xlabel('time (s)')
    plt.ylabel('log loss value')
    plt.title('Convergence of Algorithms (logarithm)')
    plt.savefig('img/log_curves.pdf')
    plt.show()

    hs = torch.stack(hs).detach()
    xs = torch.stack(xs).detach()
    print("---")
    es = torch.stack(estimated_hs_gn).detach()
    fig = plt.figure(figsize=(5, 4))
    plt.plot(ts, hs[:, 0], c='g', label='true')
    plt.plot(ts, hs[:, 1], c='g',)
    plt.plot(ts, hs[:, 2], c='g',)
    plt.plot(ts, hs[:, 3], c='g',)
    plt.plot(ts, es[:, 0], c='b', label='estimated')
    plt.plot(ts, es[:, 1], c='b',)
    plt.plot(ts, es[:, 2], c='b',)
    plt.plot(ts, es[:, 3], c='b',)
    plt.plot(ts, xs[:, 0], c='r', linestyle='--', label='observed')
    plt.plot(ts, xs[:, 1], c='r', linestyle='--')
    plt.title('Estimated GN')
    plt.xlabel('time (observations)')
    plt.ylabel('position')
    plt.legend()

    plt.savefig('img/estimated_gn.pdf')
    plt.show()

    print("---")
    es = torch.stack(estimated_hs_gn).detach()
    fig = plt.figure(figsize=(5, 4))
    plt.plot(ts, hs[:, 0], c='g', label='true')
    plt.plot(ts, hs[:, 2], c='g',)
    plt.plot(ts, es[:, 0], c='b', label='estimated')
    plt.plot(ts, es[:, 2], c='b',)
    plt.plot(ts, xs[:, 0], c='r', linestyle='--', label='observed')
    plt.plot(ts, xs[:, 1], c='r', linestyle='--')
    plt.title('Estimated GN (positions)')
    plt.xlabel('time (observations)')
    plt.ylabel('position')
    plt.legend()

    plt.savefig('img/estimated_gn_pos.pdf')
    plt.show()

    print("---")
    es = torch.stack(estimated_hs_gn).detach()
    fig = plt.figure(figsize=(5, 4))
    plt.plot(ts, hs[:, 1], c='g', label='true')
    plt.plot(ts, hs[:, 3], c='g',)
    plt.plot(ts, es[:, 1], c='b', label='estimated')
    plt.plot(ts, es[:, 3], c='b',)
    plt.title('Estimated GN (velocities)')
    plt.xlabel('time (observations)')
    plt.ylabel('position')
    plt.legend()

    plt.savefig('img/estimated_gn_vel.pdf')
    plt.show()



    print("---")
    es = torch.stack(estimated_hs_sgd).detach()
    fig = plt.figure(figsize=(5, 4))
    plt.plot(ts, hs[:, 0], c='g', label='true')
    plt.plot(ts, hs[:, 1], c='g',)
    plt.plot(ts, hs[:, 2], c='g',)
    plt.plot(ts, hs[:, 3], c='g',)
    plt.plot(ts, es[:, 0], c='b', label='estimated')
    plt.plot(ts, es[:, 1], c='b',)
    plt.plot(ts, es[:, 2], c='b',)
    plt.plot(ts, es[:, 3], c='b',)
    plt.plot(ts, xs[:, 0], c='r', linestyle='--', label='observed')
    plt.plot(ts, xs[:, 1], c='r', linestyle='--')
    plt.title('Estimated SGD')
    plt.xlabel('time (observations)')
    plt.ylabel('position')
    plt.legend()
    
    plt.savefig('img/estimated_sgd.pdf')
    plt.show()

    print("---")
    es = torch.stack(estimated_hs_sgd).detach()
    fig = plt.figure(figsize=(5, 4))
    plt.plot(ts, hs[:, 0], c='g', label='true')
    plt.plot(ts, hs[:, 2], c='g',)
    plt.plot(ts, es[:, 0], c='b', label='estimated')
    plt.plot(ts, es[:, 2], c='b',)
    plt.plot(ts, xs[:, 0], c='r', linestyle='--', label='observed')
    plt.plot(ts, xs[:, 1], c='r', linestyle='--')
    plt.title('Estimated SGD (positions)')
    plt.xlabel('time (observations)')
    plt.ylabel('position')
    plt.legend()

    plt.savefig('img/estimated_sgd_pos.pdf')
    plt.show()

    print("---")
    es = torch.stack(estimated_hs_sgd).detach()
    fig = plt.figure(figsize=(5, 4))
    plt.plot(ts, hs[:, 1], c='g', label='true')
    plt.plot(ts, hs[:, 3], c='g',)
    plt.plot(ts, es[:, 1], c='b', label='estimated')
    plt.plot(ts, es[:, 3], c='b',)
    plt.title('Estimated SGD (velocities)')
    plt.xlabel('time (observations)')
    plt.ylabel('position')
    plt.legend()

    plt.savefig('img/estimated_sgd_vel.pdf')
    plt.show()

    print("---")
    es = torch.stack(estimated_hs_lbfgs).detach()
    fig = plt.figure(figsize=(5, 4))
    plt.plot(ts, hs[:, 0], c='g', label='true')
    plt.plot(ts, hs[:, 1], c='g',)
    plt.plot(ts, hs[:, 2], c='g',)
    plt.plot(ts, hs[:, 3], c='g',)
    plt.plot(ts, es[:, 0], c='b', label='estimated')
    plt.plot(ts, es[:, 1], c='b',)
    plt.plot(ts, es[:, 2], c='b',)
    plt.plot(ts, es[:, 3], c='b',)
    plt.plot(ts, xs[:, 0], c='r', linestyle='--', label='observed')
    plt.plot(ts, xs[:, 1], c='r', linestyle='--')
    plt.title('Estimated L-BFGS')
    plt.xlabel('time (observations)')
    plt.ylabel('position')
    plt.legend()

    plt.savefig('img/estimated_lbfgs.pdf')
    plt.show()

    print("---")
    es = torch.stack(estimated_hs_lbfgs).detach()
    fig = plt.figure(figsize=(5, 4))
    plt.plot(ts, hs[:, 0], c='g', label='true')
    plt.plot(ts, hs[:, 2], c='g',)
    plt.plot(ts, es[:, 0], c='b', label='estimated')
    plt.plot(ts, es[:, 2], c='b',)
    plt.plot(ts, xs[:, 0], c='r', linestyle='--', label='observed')
    plt.plot(ts, xs[:, 1], c='r', linestyle='--')
    plt.title('Estimated L-BFGS (positions)')
    plt.xlabel('time (observations)')
    plt.ylabel('position')
    plt.legend()
    
    plt.savefig('img/estimated_lbfgs_pos.pdf')
    plt.show()

    print("---")
    es = torch.stack(estimated_hs_lbfgs).detach()
    fig = plt.figure(figsize=(5, 4))
    plt.plot(ts, hs[:, 1], c='g', label='true')
    plt.plot(ts, hs[:, 3], c='g',)
    plt.plot(ts, es[:, 1], c='b', label='estimated')
    plt.plot(ts, es[:, 3], c='b',)
    plt.title('Estimated L-BFGS (velocities)')
    plt.xlabel('time (observations)')
    plt.ylabel('position')
    plt.legend()

    plt.savefig('img/estimated_lbfgs_vel.pdf')
    plt.show()

    
    # print("---")
    # print("Simulated last state from Initial State:", )
    # print("Simulated last state from Estimated State:", )
    # print("---")

    # print("Difference (Y_estimated - Y_0):", y_estimated - y_0)
    # print("Difference (Estimated State - Initial State):",
    #       estimated_state - initial_state)

    # assert optimized_likelihood > initial_likelihood, (
    #     "Optimization failed to improve log-likelihood. "
    #     f"Initial: {initial_likelihood}, Optimized:"
    #     f" {optimized_likelihood}."
    # )


def animate(traj, l1, l2):
    # Constants (can be moved to a config file)
    L = 2.0  # max combined length for animation
    """Animates the double pendulum trajectory."""
    x1 =  l1 * traj[:, 0].sin()
    y1 = -l1 * traj[:, 0].cos()
    x2 =  l2 * traj[:, 2].sin() + x1
    y2 = -l2 * traj[:, 2].cos() + y1

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


def animate_2(traj_1, traj_2, l1, l2):
    # Constants (can be moved to a config file)
    L = 2.0  # max combined length for animation
    """Animates the double pendulum trajectory."""
    x1_1 =  l1 * traj_1[:, 0].sin()
    y1_1 = -l1 * traj_1[:, 0].cos()
    x2_1 =  l2 * traj_1[:, 2].sin() + x1_1
    y2_1 = -l2 * traj_1[:, 2].cos() + y1_1

    x1_2 =  l1 * traj_2[:, 0].sin()
    y1_2 = -l1 * traj_2[:, 0].cos()
    x2_2 =  l2 * traj_2[:, 2].sin() + x1_2
    y2_2 = -l2 * traj_2[:, 2].cos() + y1_2

    fig = plt.figure(figsize=(5,4))
    plt.plot(traj_1[:, 0], c='r', label='A')
    plt.plot(traj_1[:, 2], c='r')
    plt.plot(traj_2[:, 0], c='g', label='B')
    plt.plot(traj_2[:, 2], c='g')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('position coordinates')
    plt.title('Positions Coordinates Over Time')
    plt.savefig('img/positions_over_time.pdf')
    plt.show()

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.))
    ax.set_aspect('equal')
    ax.grid()

    line_1, = ax.plot([], [], 'o-', lw=2, label='A')
    trace_1, = ax.plot([], [], '.-', lw=1, ms=2)
    line_2, = ax.plot([], [], 'o-', lw=2, label='B')
    trace_2, = ax.plot([], [], '.-', lw=1, ms=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def animation_frame(i):
        thisx_1 = [0, x1_1[i], x2_1[i]]
        thisy_1 = [0, y1_1[i], y2_1[i]]
        thisx_2 = [0, x1_2[i], x2_2[i]]
        thisy_2 = [0, y1_2[i], y2_2[i]]
        history_x_1 = x2_1[:i]
        history_y_1 = y2_1[:i]
        history_x_2 = x2_2[:i]
        history_y_2 = y2_2[:i]
        line_1.set_data(thisx_1, thisy_1)
        line_2.set_data(thisx_2, thisy_2)
        trace_1.set_data(history_x_1, history_y_1)
        trace_2.set_data(history_x_2, history_y_2)
        time_text.set_text(time_template % (i * dt.numpy()))
        return line_1, line_2, trace_1, trace_2, time_text

    ani = animation.FuncAnimation(
        fig, animation_frame, len(traj_1), interval=dt.numpy() * 1000,
        blit=True,
    )
    plt.legend()

    plt.show()




if __name__ == '__main__':
    m1 = torch.tensor(1.0)
    m2 = torch.tensor(1.0)
    l1 = torch.tensor(1.0)
    l2 = torch.tensor(1.0)
    g = torch.tensor(9.8)
    dt = torch.tensor(0.01)
    step_size = 100
    steps = 5
    initial_state = torch.tensor([
        np.pi/2, 0.0, np.pi / 2, 0.0
    ], dtype=torch.float64)

    traj_1 = trajectory(
        initial_state, dt, 10*step_size,
        m1, m2, l1, l2, g
    ).detach()
    traj_2 = trajectory(
        initial_state + torch.randn_like(initial_state)/100, dt, 10*step_size,
        m1, m2, l1, l2, g
    ).detach()
#     animate(traj_1, l1, l2)
#     animate(traj_2, l1, l2)
#     animate_2(traj_1, traj_2, l1, l2)
# 
    test_gradient_finite_difference_agreement(
        initial_state, dt, step_size,
        m1, m2, l1, l2, g,
    )
    test_log_likelihood_evaluates(
        initial_state, dt, step_size,
        m1, m2, l1, l2, g,
    )
    test_optimization_improves_likelihood(
        initial_state, dt, step_size, steps,
        m1, m2, l1, l2, g,
    )
