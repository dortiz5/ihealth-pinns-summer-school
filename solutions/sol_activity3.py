# Import NumPy for numerical operations
import numpy as np
# Import PyTorch for building and training neural networks
import torch
import torch.nn as nn
import torch.optim as optim
# Import Matplotlib for plotting
import matplotlib.pyplot as plt
# Import a utility module 
import utils 
# Import the time module to time our training process
import time
# Ignore Warning Messages
import warnings
warnings.filterwarnings("ignore")

# torch definition of pi number
torch.pi = torch.acos(torch.zeros(1)).item() * 2


# Function to calculate the relative l2 error
def relative_l2_error(u_num, u_ref):
    # Calculate the L2 norm of the difference
    l2_diff = torch.norm(u_num - u_ref, p=2)
    
    # Calculate the L2 norm of the reference
    l2_ref = torch.norm(u_ref, p=2)
    
    # Calculate L2 relative error
    relative_l2 = l2_diff / l2_ref
    return relative_l2


# Function to plot the solutions
def plot_comparison(u_true, u_pred, loss, k_evol):
    
    # Convert tensors to numpy arrays for plotting
    u_pred_np = u_pred.detach().numpy()

    # Create a figure with 4 subplots
    fig1, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the true values
    im1 = axs[0].imshow(u_true, extent=[-1,1,1,0])
    axs[0].set_title('Analytic solution for diffusion')
    axs[0].set_xlabel(r'$x$')
    axs[0].set_ylabel(r'$t$')
    fig1.colorbar(im1, spacing='proportional',
                            shrink=0.5, ax=axs[0])

    # Plot the predicted values
    im2 = axs[1].imshow(u_pred_np, extent=[-1,1,1,0])
    axs[1].set_title('PINN solution for diffusion')
    axs[1].set_xlabel(r'$x$')
    axs[1].set_ylabel(r'$t$')
    fig1.colorbar(im2, spacing='proportional',
                            shrink=0.5, ax=axs[1])
    # Display the plot
    plt.tight_layout()
    plt.show()


    # Plot the loss values recorded during training
    # Create a figure with 2 subplots
    fig2, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Plot the difference between the predicted and true values
    axs[0].plot(k_evol, label="PINN estimate")
    axs[0].hlines(1, 0, len(k_evol), label="True value", color="tab:green")
    axs[0].set_title(r"$\kappa$ evolution")
    axs[0].set_xlabel("Iteration")
    
    axs[1].plot(loss)
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Loss')
    axs[1].set_yscale('log')
    axs[1].set_xscale('log')
    axs[1].set_title('Training Progress')
    axs[1].grid(True)

    # Display the plot
    plt.tight_layout()
    plt.show()
    
    
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
                        retain_graph=True,  
                        )[0]
    
    
#%% --------------------------------------------------------------------------- 
# Number of samples in x and t
dom_samples = 100

# Function for the diffusion analytical solution
def analytic_diffusion(x,t):
    u = np.exp(-t)*np.sin(np.pi*x)
    return u

# spatial domain
x = np.linspace(-1, 1, dom_samples)
# temporal domain
t = np.linspace(0, 1, dom_samples)

# Domain mesh
X, T = np.meshgrid(x, t)
U = analytic_diffusion(X, T)

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(U, extent=[-1,1,1,0])
ax.set_title('Analytic solution for diffusion')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$t$') 
ax.legend(loc='lower left', frameon=False)
plt.tight_layout()
plt.show()


#%% --------------------------------------------------------------------------- 
from scipy.stats import qmc
# LHS sampling strategy
sampler = qmc.LatinHypercube(d=2)
sample = sampler.random(n=100)

# lower and upper boundas of the domain
l_bounds = [-1, 0]
u_bounds = [ 1, 1]
domain_xt = qmc.scale(sample, l_bounds, u_bounds)

# torch tensors
x_ten = torch.tensor(domain_xt[:, 0], requires_grad = True).float().reshape(-1,1)
t_ten = torch.tensor(domain_xt[:, 1], requires_grad = True).float().reshape(-1,1)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(domain_xt[:, 0],domain_xt[:, 1], label = 'PDE collocation points')
ax.set_title('Collocation points')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$t$') 
ax.legend(loc='lower left')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


#%% --------------------------------------------------------------------------- 
# evaluate sample points in analytical function
x_np = x_ten.detach().numpy()
t_np = t_ten.detach().numpy()
u_true = analytic_diffusion(x_np,t_np).reshape(1, -1)
u_observ = u_true + np.random.normal(0,0.01,len(x_np))
# convert observations in Pytorch tensors
u_observ_t = torch.tensor(u_observ, requires_grad = True).float().reshape(-1,1)


#%% --------------------------------------------------------------------------- 
torch.manual_seed(123)

# training parameters
hidden_layers = [2, 20, 20, 20, 1]
learning_rate = 0.001
training_iter = 40000


#%% --------------------------------------------------------------------------- 
# Define a loss function (Mean Squared Error) for training the network
MSE_func = nn.MSELoss()

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
        self.init_params
        
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
u_pinn = NeuralNetwork(hidden_layers)
nparams = sum(p.numel() for p in u_pinn.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {nparams}')


# treat k as a learnable parameter
kappa = torch.nn.Parameter(torch.ones(1, requires_grad=True)*2)
kappas = []

# add k to the optimiser
# Define an optimizer (Adam) for training the network
optimizer = optim.Adam(list(u_pinn.parameters())+[kappa], lr=0.001, 
                       betas= (0.9,0.999), eps = 1e-8)

#%% --------------------------------------------------------------------------- 

def PINN_diffusion_Loss(forward_pass, x_ten, t_ten,
             lambda1 = 1, lambda2 = 1):

    # ANN output, first and second derivatives
    domain = torch.cat([t_ten, x_ten], dim = 1)
    u = forward_pass(domain)
    u_t = grad(u, t_ten)
    u_x = grad(u, x_ten)
    u_xx = grad(u_x, x_ten)
    
    # PDE loss definition
    f_pde = u_t - kappa*u_xx + torch.exp(-t_ten)*(torch.sin(np.pi*x_ten) 
                                        -(torch.pi**2)*torch.sin(np.pi*x_ten))
    PDE_loss = lambda1 * MSE_func(f_pde, torch.zeros_like(f_pde)) 
    
    # Data loss
    data_loss = lambda2 * MSE_func(u, u_observ_t)
    
    return PDE_loss + data_loss
    
# Initialize a list to store the loss values
loss_values = []

# Start the timer
start_time = time.time()

# Training the neural network
for i in range(training_iter):
    
    optimizer.zero_grad()   # clear gradients for next train

    # input x and predict based on x
    loss = PINN_diffusion_Loss(u_pinn, x_ten, t_ten)
    
    # Append the current loss value to the list
    loss_values.append(loss.item())
    kappas.append(kappa.item())

    if i % 1000 == 0:  # print every 100 iterations
        print(f"Iteration {i}: Loss {loss.item()}")
    
    loss.backward() # compute gradients (backpropagation)
    optimizer.step() # update the ANN weigths

# Stop the timer and calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training time: {elapsed_time} seconds")


#%% --------------------------------------------------------------------------- 

X_ten = torch.tensor(X).float().reshape(-1, 1)
T_ten = torch.tensor(T).float().reshape(-1, 1)
domain_ten = torch.cat([T_ten, X_ten], dim = 1)
U_pred = u_pinn(domain_ten).reshape(dom_samples,dom_samples)

U_true = torch.tensor(U).float()
print(f'Relative error: {relative_l2_error(U_pred, U_true)}')

plot_comparison(U, U_pred, loss_values, kappas)