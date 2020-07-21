import numpy as np
import scipy.optimize
import sparse_autoencoder
import display_network



############################################################################################
# ================= Loading Image Dataset ================================================ #
############################################################################################

from mlxtend.data import mnist_data
X, y = mnist_data()

# images.shape : (num_rows * num_cols, num_images)
images = X.T
# set height and width of a single image
num_rows = 28
num_cols = 28

assert num_rows * num_cols == images.shape[0]  
num_images = images.shape[1]
input_size = num_rows * num_cols
images = images.astype(np.float64) / 255

############################################################################################
############################################################################################



############################################################################################
# ==================== Setting Hyper Parameters ========================================== #
############################################################################################

# Number of hidden layer neurons
hidden_size = 196
# Sparsity parameter (desired average activation of each hidden layer neuron)
ρ = 0.1
# Weight decay parameter
λ = 3e-3
# Weight of sparsity penalty term
β = 3

############################################################################################
############################################################################################



############################################################################################
# =============================== Train ================================================== #
############################################################################################

# Initialize parameters of the model
θ = sparse_autoencoder.initialize_params(hidden_size, input_size)
# Declare cost function
J = lambda θ: sparse_autoencoder.calc_cost_n_gradient(θ, input_size, hidden_size,
                                                      λ, ρ, β, images)
# Set training options
options_ = {'maxiter': 1000, 'disp': True}
# Minimize cost function J by modifying parameters θ using L-BFGS-B optimization algo
result = scipy.optimize.minimize(J, θ, method='L-BFGS-B', jac=True, options=options_)
# Get optimized parameter vector
opt_θ = result.x

print(result)

############################################################################################
############################################################################################



############################################################################################
# =============================== Visualization ========================================== #
############################################################################################

# Slice fan-in weight vector of the hidden layer from optimized parameter vector 
W1 = opt_θ[0:hidden_size * input_size].reshape(hidden_size, input_size).transpose()
# Reshape the fan-in weight vector to obtain features learned by each hidden layer neuron 
display_network.display_network(W1, filename='weights-'+str(hidden_size)+'-'+str(ρ)+'-'+str(λ)+'-'+str(β)+'.png')

############################################################################################
############################################################################################
