# Import NumPy for numerical operations
import numpy as np
# Import PyTorch for building and training neural networks
import torch
import torch.nn as nn
import torch.optim as optim
# Import Matplotlib for plotting
import matplotlib.pyplot as plt
import matplotlib as mlp
# Import the time module to time our training process
import time
# Ignore Warning Messages
import warnings
warnings.filterwarnings("ignore")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Actualización de los parámetros de Matplotlib
gray = '#5c5c5c' #'#5c5c5c' '000'
mlp.rcParams.update(
    {
        "image.cmap" : 'viridis', # plasma, inferno, magma, cividis
        "text.color" : gray,
        "xtick.color" :gray,
        "ytick.color" :gray,
        "axes.labelcolor" : gray,
        "axes.edgecolor" :gray,
        "axes.spines.right" : False,
        "axes.spines.top" : False,
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        
        'font.size' : 15,
        'interactive': False,
        "font.family": 'sans-serif',
        "legend.loc" : 'best',
        'text.usetex': False,
        'mathtext.fontset': 'stix',
    }
)

# Util function to calculate the signal-to-noise ratio
def calculate_snr(signal, noise):    
    # Ensure numpy arrays
    signal, noise = np.array(signal), np.array(noise)
    
    # Calculate the power of the signal and the noise
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    
    # Calculate the SNR in decibels (dB)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Util function to calculate the relative l2 error
def relative_l2_error(u_num, u_ref):
    # Calculate the L2 norm of the difference
    l2_diff = torch.norm(u_num - u_ref, p=2)
    
    # Calculate the L2 norm of the reference
    l2_ref = torch.norm(u_ref, p=2)
    
    # Calculate L2 relative error
    relative_l2 = l2_diff / l2_ref
    return relative_l2

# Util function to plot the solutions
def plot_comparison(time, theta_true, theta_pred, loss):
    
    # Convert tensors to numpy arrays for plotting
    t_np = time.detach().cpu().data.numpy()
    theta_pred_np = theta_pred.detach().cpu().data.numpy()

    # Create a figure with 2 subplots
    _, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the true and predicted values
    axs[0].plot(t_np, theta_true, label = r'$\theta(t)$ (numerical solution)')
    axs[0].plot(t_np, theta_pred_np, label = r'$\theta_{pred}(t)$ (predicted solution) ')
    axs[0].set_title('Angular displacement Numerical Vs. Predicted')
    axs[0].set_xlabel(r'Time $(s)$')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_ylim(-1,1.3)
    axs[0].legend(loc='best', frameon=False)


    # Plot the difference between the predicted and true values
    difference = np.abs(theta_true.reshape(-1,1) - theta_pred_np.reshape(-1,1))
    axs[1].plot(t_np, difference)
    axs[1].set_title('Absolute Difference')
    axs[1].set_xlabel(r'Time $(s)$')
    axs[1].set_ylabel(r'$|\theta(t) - \theta_{pred}(t)|$')
    # Display the plot
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.show()

    # Plot the loss values recorded during training
    # Create a figure with 1 subplots
    _, axs = plt.subplots(1, 1, figsize=(6, 3))
    axs.plot(loss)
    axs.set_xlabel('Iteration')
    axs.set_ylabel('Loss')
    axs.set_yscale('log')
    axs.set_xscale('log')
    axs.set_title('Training Progress')
    axs.grid(True)

    # Display the plot
    plt.tight_layout()
    plt.show()

# Util function to calculate tensor gradients with autodiff
def grad(outputs, inputs):
    """Computes the partial derivative of an output with respect 
    to an input.
    Args:
        outputs: (N, 1) tensor
        inputs: (N, D) tensor
    """
    return torch.autograd.grad(outputs, inputs, 
                        grad_outputs=torch.ones_like(outputs), 
                        create_graph=True,
                        )[0]
    
#%% --------------------------------------------------------------------------- 
g = 9.81  # gravity acceleration (m/s^2)
L = 1.0   # Pendulum's rod length (m)
theta0 = np.pi / 4  # Initial condition (Position in rads)
omega0 = 0.0        # Initial angular speed (rad/s)
sample_freq = 100   # sample rate 100Hz

# Simulation time
t_span = (0, 10)  # from 0 to 10 seconds
t_eval = np.linspace(t_span[0], t_span[1], sample_freq*t_span[1])  # Points to be evaluated

# We define the system of coupled ODEs
def pendulum(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta)
    return [dtheta_dt, domega_dt]

# Initial conditions
y0 = [theta0, omega0]


#%% --------------------------------------------------------------------------- 
from scipy.integrate import solve_ivp

# Solve the initial value problem using Runge-Kutta 4th order
num_sol = solve_ivp(pendulum, t_span, y0, t_eval=t_eval, method='RK45')

# We extract the solutions
theta_num = num_sol.y[0]
omega_num = num_sol.y[1]

# We graph the results
plt.figure(figsize=(12, 6))
plt.plot(t_eval, theta_num, label=r'Angular Displacement $\theta(t)[rad]$')
plt.plot(t_eval, omega_num, label=r'Angular Velocity $\omega(t)[rad/s]$')
plt.xlabel(r'Time $[s]$')
plt.ylim(-2.5,3.3)
plt.legend(loc='best', frameon=False)
plt.title('Nonlinear Pendulum Solution')
plt.grid(False)
plt.tight_layout()
plt.show()


#%% --------------------------------------------------------------------------- 
# Add gaussian noise
std_deviation = 0.05
noise = np.random.normal(0,std_deviation,theta_num.shape[0])
theta_noisy = theta_num + noise
print(f'SNR: {calculate_snr(theta_noisy, noise):.4f} dB')

# Resample and cut to 2.5s
resample = 5          # resample 
cut_time = int(2.5*sample_freq)  # 2.5s times 100Hz

theta_data = theta_noisy[:cut_time:resample]
t_data = t_eval[:cut_time:resample]

# We graph the observed data
plt.figure(figsize=(12, 6))
plt.plot(t_eval, theta_num, label=r'Angular Displacement (model) $\theta(t)$ ')
plt.plot(t_data, theta_data, label=r'Training data (measures) $\theta_{data}(t)$ ')
plt.xlabel(r'Time $[s]$')
plt.ylabel(r'Angular displacement $[rad]$')
plt.ylim(-1,1.3)
plt.legend(loc='best', frameon=False)
plt.title('Training data')
plt.grid(False)
plt.show()


#%% --------------------------------------------------------------------------- 
torch.manual_seed(123)

# training parameters
hidden_layers = [1, 20, 20, 20, 1]
learning_rate = 0.001
training_iter = 50000


#%% --------------------------------------------------------------------------- 
# Define a loss function (Mean Squared Error) for training the network
MSE_func = nn.MSELoss()

# Convert the NumPy arrays to PyTorch tensors and add an extra dimension
# test time Numpy array to Pytorch tensor
t_test = torch.tensor(t_eval, device=device, requires_grad=True).view(-1,1).float()
# train time Numpy array to Pytorch tensor
t_data = torch.tensor(t_data, device=device, requires_grad=True).view(-1,1).float()
# Numerical theta to test Numpy array to pytorch tensor 
theta_test = torch.tensor(theta_num, device=device, requires_grad=True).view(-1,1).float()
# Numerical theta to train Numpy array to pytorch tensor 
theta_data = torch.tensor(theta_data, device=device, requires_grad=True).view(-1,1).float()

# Define a neural network class with user defined layers and neurons
class NeuralNetwork(nn.Module):
    
    def __init__(self, hlayers):
        super(NeuralNetwork, self).__init__()
        
        layers = []
        for i in range(len(hlayers[:-2])):
            layers.append(nn.Linear(hlayers[i], hlayers[i+1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hlayers[-2], hlayers[-1]))
        
        self.layers = nn.Sequential(*layers)
        self.init_params()
        
    def init_params(self):
        """Xavier Glorot parameter initialization of the Neural Network
        """
        def init_normal(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight) # Xavier
        self.apply(init_normal)

    def forward(self, x):
        return self.layers(x)
    
    
#%% --------------------------------------------------------------------------- 
# Create an instance of the neural network 
theta_ann = NeuralNetwork(hidden_layers).to(device)
nparams = sum(p.numel() for p in theta_ann.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {nparams}')

# Define an optimizer (Adam) for training the network
optimizer = optim.Adam(theta_ann.parameters(), lr=learning_rate, 
                       betas= (0.9,0.999), eps = 1e-8)


#%% --------------------------------------------------------------------------- 
def NeuralNetworkLoss(forward_pass, t, theta_data, lambda1 = 1):
    
    theta_nn = forward_pass(t)
    data_loss = lambda1 * MSE_func(theta_nn, theta_data)
    
    return  data_loss
    
# Initialize a list to store the loss values
loss_values_ann = []

# Start the timer
start_time = time.time()

# Training the neural network
for i in range(training_iter):
    
    optimizer.zero_grad()   # clear gradients for next train

    # input x and predict based on x
    loss = NeuralNetworkLoss(theta_ann,
                             t_data,
                             theta_data)    # must be (1. nn output, 2. target)
    
    # Append the current loss value to the list
    loss_values_ann.append(loss.item())
    
    if i % 1000 == 0:  # print every 100 iterations
        print(f"Iteration {i}: Loss {loss.item()}")
    
    loss.backward() # compute gradients (backpropagation)
    optimizer.step() # update the ANN weigths

# Stop the timer and calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training time: {elapsed_time} seconds")



#%% --------------------------------------------------------------------------- 
theta_pred_ann = theta_ann(t_test).to(device)

print(f'Relative error: {relative_l2_error(theta_pred_ann, theta_test)}')

plot_comparison(t_test, theta_num, theta_pred_ann, loss_values_ann)

#%% --------------------------------------------------------------------------- 
# Create an instance of the neural network
theta_pinn = NeuralNetwork(hidden_layers).to(device)
nparams = sum(p.numel() for p in theta_pinn.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {nparams}')

# Define an optimizer (Adam) for training the network
optimizer = optim.Adam(theta_pinn.parameters(), lr=learning_rate, 
                       betas= (0.9,0.999), eps = 1e-8)


#%% --------------------------------------------------------------------------- 
# Define t = 0 for boundary an initial conditions 
t0 = torch.tensor(0., requires_grad=True).view(-1,1)

# HINT: 
def PINNLoss(forward_pass, t_phys, t_data, theta_data, 
             lambda1 = 1, lambda2 = 1, lambda3 = 1, lambda4 = 1):

    # ANN output, first and second derivatives
    theta_pinn1 = forward_pass(t_phys)
    theta_pinn_dt = grad(theta_pinn1, t_phys)
    theta_pinn_ddt = grad(theta_pinn_dt, t_phys)
    
    f_ode = theta_pinn_ddt + (g/L) * torch.sin(theta_pinn1)
    ODE_loss = lambda1 * MSE_func(f_ode, torch.zeros_like(f_ode)) 
    
    g_ic = forward_pass(t0)
    IC_loss = lambda2 * MSE_func(g_ic, torch.ones_like(g_ic)*theta0)
    
    h_bc = grad(forward_pass(t0),t0)
    BC_loss = lambda3 * MSE_func(h_bc, torch.zeros_like(h_bc))
    
    theta_nn2 = forward_pass(t_data)
    data_loss = lambda4 * MSE_func(theta_nn2, theta_data)
    
    return ODE_loss + IC_loss + BC_loss + data_loss
    
# Initialize a list to store the loss values
loss_values_pinn = []

# Start the timer
start_time = time.time()

# Training the neural network
for i in range(training_iter):
    
    optimizer.zero_grad()   # clear gradients for next train

    # input x and predict based on x
    loss = PINNLoss(theta_pinn, t_test, t_data, theta_data)
    
    # Append the current loss value to the list
    loss_values_pinn.append(loss.item())
    
    if i % 1000 == 0:  # print every 100 iterations
        print(f"Iteration {i}: Loss {loss.item()}")
    
    loss.backward() # compute gradients (backpropagation)
    optimizer.step() # update the ANN weigths

# Stop the timer and calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training time: {elapsed_time} seconds")



#%% --------------------------------------------------------------------------- 
theta_pred_pinn = theta_pinn(t_test)

print(f'Relative error: {relative_l2_error(theta_pred_pinn, theta_test)}')

plot_comparison(t_test, theta_num, theta_pred_pinn, loss_values_pinn)