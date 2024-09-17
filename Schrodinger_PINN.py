# Import necessary libraries
import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite  # For Hermite polynomials
import matplotlib.animation as animation

# Define the neural network model
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        # Activation function
        self.activation = nn.Tanh()
        # Define neural network layers
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
    
    def forward(self, x, t):
        # Concatenate x and t to create input
        inputs = torch.cat((x, t), dim=1)
        # Forward pass through the network
        for i in range(len(self.layers) - 1):
            inputs = self.activation(self.layers[i](inputs))
        # Last layer without activation
        outputs = self.layers[-1](inputs)
        return outputs

# Define the physics-informed loss function
class SchrodingerPINN_HO:
    def __init__(self, model):
        self.model = model

    def net_psi(self, x, t):
        # Enable gradient computation
        x.requires_grad = True
        t.requires_grad = True

        # Forward pass through the neural network
        psi = self.model(x, t)
        psi_r = psi[:, 0:1]  # Real part
        psi_i = psi[:, 1:2]  # Imaginary part
        psi_c = psi_r + 1j * psi_i  # Complex wave function

        # First-order time derivative
        psi_c_t = torch.autograd.grad(psi_c, t,
                                      grad_outputs=torch.ones_like(psi_c),
                                      create_graph=True)[0]

        # First-order spatial derivative
        psi_c_x = torch.autograd.grad(psi_c, x,
                                      grad_outputs=torch.ones_like(psi_c),
                                      create_graph=True)[0]
        # Second-order spatial derivative
        psi_c_xx = torch.autograd.grad(psi_c_x, x,
                                       grad_outputs=torch.ones_like(psi_c_x),
                                       create_graph=True)[0]

        return psi_c, psi_c_t, psi_c_xx

    def loss_function(self, x_f, t_f, x_i, t_i, psi_i_exact):
        # Compute physics loss (TDSE residual)
        psi_c, psi_c_t, psi_c_xx = self.net_psi(x_f, t_f)
        V = 0.5 * x_f ** 2  # Harmonic oscillator potential
        residual = psi_c_t + 1j * (0.5 * psi_c_xx - V * psi_c)
        physics_loss = torch.mean(torch.abs(residual) ** 2)

        # Compute initial condition loss
        psi_i_pred = self.model(x_i, t_i)
        ic_loss = torch.mean((psi_i_pred - psi_i_exact) ** 2)

        # Total loss
        total_loss = physics_loss + ic_loss
        return total_loss

# Function to compute the initial wave function for arbitrary n
def psi_initial(x, n):
    # Convert x to NumPy array for compatibility with scipy
    x_np = x.detach().cpu().numpy().flatten()
    # Compute the Hermite polynomial H_n(x)
    Hn = hermite(n)
    Hn_x = Hn(x_np)
    # Normalization constant
    norm_const = (1 / np.pi ** 0.25) * (1 / np.sqrt(2 ** n * math.factorial(n)))
    # Compute the initial wave function
    psi_n = norm_const * np.exp(-x_np ** 2 / 2) * Hn_x
    # Convert back to torch tensor
    psi_n_tensor = torch.tensor(psi_n, dtype=torch.float32).reshape(-1, 1)
    return psi_n_tensor

# Set the quantum numbers to solve for
n_values = [0,1]  # List of n values

# Loop over each n
for n in n_values:
    print(f"\nSolving for n = {n}")

    # Domain boundaries
    x_min, x_max = -5.0, 5.0
    t_min, t_max = 0.0, 2.0

    # Number of training points
    N_f = 20000  # Collocation points for physics loss
    N_i = 2000   # Initial condition points

    # Collocation points inside the domain (physics loss)
    x_f = torch.FloatTensor(N_f, 1).uniform_(x_min, x_max).requires_grad_(True)
    t_f = torch.FloatTensor(N_f, 1).uniform_(t_min, t_max).requires_grad_(True)

    # Initial condition points (t = 0)
    x_i = torch.FloatTensor(N_i, 1).uniform_(x_min, x_max).requires_grad_(True)
    t_i = torch.zeros_like(x_i).requires_grad_(True)

    # Compute the initial wave function at x_i
    psi_i_real = psi_initial(x_i, n)
    psi_i_imag = torch.zeros_like(psi_i_real)
    psi_i_exact = torch.cat([psi_i_real, psi_i_imag], dim=1)  # Real and imaginary parts

    # Define the neural network architecture
    layers = [2, 256,64,32, 2]  # Input: x, t; Output: Re(psi), Im(psi)
    model = PINN(layers)

    # Instantiate the PINN class
    pinn = SchrodingerPINN_HO(model)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Number of epochs for training
    epochs = 1000

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Compute the loss
        loss = pinn.loss_function(x_f, t_f, x_i, t_i, psi_i_exact)
        # Backpropagation
        loss.backward()
        # Update parameters
        optimizer.step()

        # Print loss every 500 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.5e}")

    # Evaluation points for x and t
    x_eval = np.linspace(x_min, x_max, 200)
    t_eval = np.linspace(t_min, t_max, 100)
    X, T = np.meshgrid(x_eval, t_eval)
    X_flat = torch.tensor(X.flatten(), dtype=torch.float32).reshape(-1, 1)
    T_flat = torch.tensor(T.flatten(), dtype=torch.float32).reshape(-1, 1)

    # Energy of the nth state
    E_n = n + 0.5  # Since omega = 1

    # Compute the initial wave function on the evaluation grid
    psi_n_x = psi_initial(X_flat, n)

    # Compute the analytical solution
    psi_exact = psi_n_x * torch.exp(-1j * E_n * T_flat)

    # Convert to numpy arrays for comparison
    psi_exact_np = psi_exact.detach().numpy()

    # Predict using the trained model
    with torch.no_grad():
        psi_pred = model(X_flat, T_flat)
        psi_pred_real = psi_pred[:, 0].numpy()
        psi_pred_imag = psi_pred[:, 1].numpy()
        psi_pred_complex = psi_pred_real + 1j * psi_pred_imag
        psi_pred_np = psi_pred_complex

    # Compute the absolute difference
    error = np.abs(psi_pred_np - psi_exact_np.squeeze())
    # Compute the L2 relative error
    l2_error = np.linalg.norm(error) / np.linalg.norm(psi_exact_np)
    print(f"L2 Relative Error for n = {n}: {l2_error:.5e}")

    # Reshape for plotting
    psi_pred_real = psi_pred_real.reshape(T.shape)
    psi_pred_imag = psi_pred_imag.reshape(T.shape)
    psi_exact_real = np.real(psi_exact_np).reshape(T.shape)
    psi_exact_imag = np.imag(psi_exact_np).reshape(T.shape)

    # Plot only the real part
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # PINN Prediction - Real Part
    im0 = ax[0].contourf(X, T, psi_pred_real, levels=100, cmap='viridis')
    fig.colorbar(im0, ax=ax[0])
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('t')
    ax[0].set_title(f'PINN Predicted Re[Ψ(x,t)] for n = {n}')

    # Analytical Solution - Real Part
    im1 = ax[1].contourf(X, T, psi_exact_real, levels=100, cmap='viridis')
    fig.colorbar(im1, ax=ax[1])
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('t')
    ax[1].set_title(f'Analytical Re[Ψ(x,t)] for n = {n}')

    plt.tight_layout()
    plt.show()  


    # Create an animation of the real and imaginary parts
    fig, ax = plt.subplots()
    line_real, = ax.plot([], [], label='Re[Ψ]', color='blue')
    line_imag, = ax.plot([], [], label='Im[Ψ]', color='red')
    ax.set_xlim(x_min, x_max)
    # Determine suitable y-limits
    y_max = max(np.max(psi_pred_real), np.max(psi_pred_imag))
    y_min = min(np.min(psi_pred_real), np.min(psi_pred_imag))
    ax.set_ylim(1.5*y_min,y_max*1.5)
    ax.legend()

    def init():
        line_real.set_data([], [])
        line_imag.set_data([], [])
        return line_real, line_imag

    def update(frame):
        t_val = t_eval[frame]
        t_tensor = torch.tensor(t_val * np.ones_like(x_eval), dtype=torch.float32).reshape(-1, 1)
        x_tensor = torch.tensor(x_eval, dtype=torch.float32).reshape(-1, 1)

        with torch.no_grad():
            psi_pred = model(x_tensor, t_tensor)
            psi_pred_real = psi_pred[:, 0].numpy()
            psi_pred_imag = psi_pred[:, 1].numpy()

        line_real.set_data(x_eval, psi_pred_real)
        line_imag.set_data(x_eval, psi_pred_imag)
        ax.set_title(f't = {t_val:.2f} for n = {n}')
        return line_real, line_imag

    ani = animation.FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True)
    plt.show()

    # Save the animation
    ani.save(f'wave_function_evolution_real_imag_n{n}.mp4', writer='ffmpeg', fps=15)
