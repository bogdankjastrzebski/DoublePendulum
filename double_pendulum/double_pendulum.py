import torch


def derivs(state, M1, M2, L1, L2, G):
    """
    Computes the derivatives for the double pendulum using PyTorch without mutating the state tensor.
    Args:
        state (torch.Tensor):
            The state vector [theta1, omega1, theta2, omega2] at time t. 
        M1 (torch.Tensor): Mass of the first pendulum.
        M2 (torch.Tensor): Mass of the second pendulum.
        L1 (torch.Tensor): Length of the first pendulum.
        L1 (torch.Tensor): Length of the second pendulum.
        G (torch.Tensor): Acceleration due to gravity.
    
    Returns:
        torch.Tensor: The derivatives [
            dtheta1_dt, domega1_dt,
            dtheta2_dt, domega2_dt, 
        ].
    """

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
    return y + derivs(y, M1, M2, L1, L2, G) * dt

def run(y, dt, n, M1, M2, L1, L2, G):
    for _ in range(n):
        y = update(y, dt, M1, M2, L1, L2, G)
    return y

def trajectory(y, dt, n, M1, M2, L1, L2, G):
    return torch.stack([
        y := update(y, dt, M1, M2, L1, L2, G)
        for _ in range(n)
    ])


def compute_grad(func, initial_state, dt, n, M1, M2, L1, L2, G):
    """Compute the Jacobian of the final state
    with respect to the initial state."""
    initial_state.requires_grad_(True)
    final_state = func(run(initial_state, dt, n, M1, M2, L1, L2, G))
    return torch.autograd.grad(
        final_state,
        initial_state,
        create_graph=False,
        retain_graph=False,
    )[0]


# if __name__ == '__main__':
M1 = torch.tensor(1.0)
M2 = torch.tensor(1.0)
L1 = torch.tensor(1.0)
L2 = torch.tensor(1.0)
G = torch.tensor(9.8)
dt = torch.tensor(0.01) 
n = 100
initial_state = torch.tensor([
    torch.pi/2, 0.0, torch.pi / 2, 0.0
], dtype=torch.float64)
def e(i, n=4):
    r = torch.zeros(n)
    r[i] = 1.0
    return r

traj = trajectory(initial_state, dt, n, M1, M2, L1, L2, G).detach()


grad0 = compute_grad(lambda x: x[0], initial_state, dt, n, M1, M2, L1, L2, G)
grad1 = compute_grad(lambda x: x[1], initial_state, dt, n, M1, M2, L1, L2, G)
grad2 = compute_grad(lambda x: x[2], initial_state, dt, n, M1, M2, L1, L2, G)
grad3 = compute_grad(lambda x: x[3], initial_state, dt, n, M1, M2, L1, L2, G)

grad = torch.stack([grad0, grad1, grad2, grad3]).T

dx = 0.01
y_h0 = run(initial_state + dt * e(0), dt, n, M1, M2, L1, L2, G)
y_h1 = run(initial_state + dt * e(1), dt, n, M1, M2, L1, L2, G)
y_h2 = run(initial_state + dt * e(2), dt, n, M1, M2, L1, L2, G)
y_h3 = run(initial_state + dt * e(3), dt, n, M1, M2, L1, L2, G)
y_0 = run(initial_state, dt, n, M1, M2, L1, L2, G)

d0 = (y_h0 - y_0)/dx
d1 = (y_h1 - y_0)/dx
d2 = (y_h2 - y_0)/dx
d3 = (y_h3 - y_0)/dx

d = torch.stack([d0, d1, d2, d3])

anim(traj, L1, L2)

def anim(traj, L1, L2):
    x1 =  L1 * traj[:, 0].sin()
    y1 = -L1 * traj[:, 0].cos()
    x2 =  L2 * traj[:, 2].sin() + x1
    y2 = -L2 * traj[:, 2].cos() + y1
    
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.))
    ax.set_aspect('equal')
    ax.grid()
    
    line, = ax.plot([], [], 'o-', lw=2)
    trace, = ax.plot([], [], '.-', lw=1, ms=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    
    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]
        history_x = x2[:i]
        history_y = y2[:i]
        line.set_data(thisx, thisy)
        trace.set_data(history_x, history_y)
        time_text.set_text(time_template % (i * dt.numpy()))
        return line, trace, time_text
    
    
    ani = animation.FuncAnimation(
        fig, animate, len(traj), interval=dt.numpy() * 1000,
        blit=True,
    )
    
    plt.show()
