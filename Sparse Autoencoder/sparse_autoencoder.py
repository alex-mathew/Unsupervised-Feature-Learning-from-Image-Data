import numpy as np


# sigmoid activation function
def σ(x):
    return 1 / (1 + np.exp(-x))


# derivate of sigmoid activation function
def delta_σ(x):
    return σ(x) * (1 - σ(x))

# KL_divergence used to compute Sparsity Penalty
def KL_divergence(x, y):
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))


def initialize_params(hidden_size, input_size):
    """
    Initialize parameters of the model

    :param visible_size: Number of input layer neurons
    :param hidden_size: Number of hidden layer neurons

    :returns θ: concatenated weights and biases of encoder & decoder
    """
    # Xavier initialization sets a layer’s weights to values chosen 
    # from a random uniform distribution that’s bounded between [-r, r]
    r = np.sqrt((6) / (hidden_size + input_size + 1))
    W1 = np.random.random((hidden_size, input_size)) * 2 * r - r
    W2 = np.random.random((input_size, hidden_size)) * 2 * r - r
    # Biases are initialized with 0
    b1 = np.zeros(hidden_size, dtype=np.float64)
    b2 = np.zeros(input_size, dtype=np.float64)
    # Concatenate weights and biases to get a weight vector
    θ = np.concatenate((W1.reshape(hidden_size * input_size),
                            W2.reshape(hidden_size * input_size),
                            b1.reshape(hidden_size),
                            b2.reshape(input_size)))
    return θ


def calc_cost_n_gradient(θ, visible_size, hidden_size,
                         λ, ρ, β, data):

    """
    Compute cost and gradient of the model parametarized with θ

    :param θ: Model Parameters (vector concating weights and biases)
    :param visible_size: Number of input layer neurons
    :param hidden_size: Number of hidden layer neurons
    :param λ: Weight Decay Parameter
    :param ρ: Sparsity Parameter
    :param β: Weight of Sparsity Penalty term
    :param data: Dataset with each data item as a column

    :returns cost, gradient: cost and gradient vector
    """

    # Slice out weight and bias of each layer from θ
    W1 = θ[0:hidden_size * visible_size].reshape(hidden_size, visible_size)
    W2 = θ[hidden_size * visible_size:2 * hidden_size * visible_size].reshape(visible_size, hidden_size)
    b1 = θ[2 * hidden_size * visible_size:2 * hidden_size * visible_size + hidden_size]
    b2 = θ[2 * hidden_size * visible_size + hidden_size:]

    # Number of training examples
    m = data.shape[1]

    # Forward propagation
    z2 = W1.dot(data) + np.tile(b1, (m, 1)).transpose()
    a2 = σ(z2)
    z3 = W2.dot(a2) + np.tile(b2, (m, 1)).transpose()
    h = σ(z3)

    # Avg. activation vector of hidden layer neurons
    rho_hat = np.sum(a2, axis=1) / m
    # Expected avg. activation vector of hidden layer neurons
    rho = np.tile(ρ, hidden_size)

    # Cost function = Reconstruction Loss +
    #                 L2 Regularization Term scaled by λ +
    #                 Sparsity Penalty Term scaled by β
    cost = np.sum((h - data) ** 2) / (2 * m) + \
           λ * (np.sum(W1 ** 2) + np.sum(W2 ** 2)) + \
           β * np.sum(KL_divergence(rho, rho_hat))

    # Gradient calculation via backpropogation
    sparsity_delta = np.tile(- rho / rho_hat + (1 - rho) / (1 - rho_hat), (m, 1)).transpose()
    delta3 = -(data - h) * delta_σ(z3)
    delta2 = (W2.transpose().dot(delta3) + β * sparsity_delta) * delta_σ(z2)
    W1grad = delta2.dot(data.transpose()) / m + λ * W1
    W2grad = delta3.dot(a2.transpose()) / m + λ * W2
    b1grad = np.sum(delta2, axis=1) / m
    b2grad = np.sum(delta3, axis=1) / m

    # Flatten and combine gradient of each parameter to a vector
    grad = np.concatenate((W1grad.reshape(hidden_size * visible_size),
                           W2grad.reshape(hidden_size * visible_size),
                           b1grad.reshape(hidden_size),
                           b2grad.reshape(visible_size)))

    return cost, grad