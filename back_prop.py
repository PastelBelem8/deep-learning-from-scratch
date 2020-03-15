import numpy as np

data = np.array([[1, 0], [1, 0]])
label_pos = 1
features_pos = [0]


def relu(x): 
    return max(0, x)


def se(y_true: float, y_pred: float):
    """Computes the Squared Error"""
    return (y_true - y_pred) ** 2

loss_funcs = {
    "se": se,
}

activation_funcs = {
    "relu": relu,
}

weights_initialization = {
    "zeros": np.zeros,
    "random": np.random.rand, 
}

# ----------------------------------------------------------------------
#                               Main Methods  
# ----------------------------------------------------------------------
def forward_propagation(
    sample: np.ndarray, 
    Ws: np.ndarray, 
    activations: tuple, 
    loss_func: callable
) -> tuple:
    """Compute a forward pass for a specific sample. 

    Given the weights ``Ws`, the ``activations`` and the loss function, 
    apply the forward propagation algorithm. Throughout this computation 
    the local derivatives at each neuron are also computed, being stored
    in a dictionary.

    Parameters
    ----------
    sample: numpy.ndarray
        Vector with the input values to start the forward pass from. 
    Ws: Iterable[numpy.ndarray]
        Iterable with the weight matrices to apply in the pre-activation
        computation. 
    activations: Iterable[callable]
        Iterable with the name of the activation functions to use 
        for each layer.  
    loss_func: str 
        Loss function to compute in the end of the forward pass.

    Returns
    -------
    tuple[float64, dict]
        Tuple with the total loss in the end of the forward pass and the 
        metadata accumulated during the forward pass, including the local 
        derivatives computed at each neuron.
    """
    X, y_true = sample[features_pos], sample[label_pos]

    Zs = [X]
    df_dZs, df_dWs = [], []

    for i, l_size in enumerate(architecture[:-1]): 
        next_layer = i-1

        Z, df_dz, df_dW = forward_propagation(Zs[i], Ws[i], activations[i])
        Zs.append(Z)
        # Local derivative with the contribution of each layer to to the 
        # output of the following layer
        df_dZs.append(df_dz)
        # Local derivative with the contribution of the weights to the 
        # output of the following layer
        df_dWs.append(df_dW)

    loss = loss_fs[loss_func](y_true, Zs[-1])
    metadata = {
        # Outputs
        "Zs": Zs, 
        # Local derivatives
        "df_dZs": df_dZs,
        "df_dWs": df_dWs, 
        # Loss
        "loss": loss,
    }
    return loss, metadata


def backward_propagation(error: np.ndarray, partial_ds: dict) -> np.ndarray: 
    """Propagates the errors backwards from the current layer `l` to the 
    previous layer `l-1`, using the chain rule to determine the contribution 
    of the components involved in computing the value of `l` to the total error. 

    Parameters
    ----------
    error : numpy.ndarray
        Represents the error contribution of the current layer `l` to the error. 
    partial_ds: dict
        Local partial derivatives involved in the forward pass from `l-1` to `l`.

    Returns
    -------
    dict    
        The partial contributions of each component to the error. 
    """
    pass
    

def train_NN(data: np.ndarray, architecture: tuple, activations: tuple, loss_func: str, weights_init: str = "random") -> dict:
    """Trains a neural network with ``architecture`` and `activation_funcs`.

    Parameters
    ----------
    data: numpy.ndarray
        Data to fit the neural network. 

    architecture: Iterable[int]
        An iterable whose dimension is n+2, where n is the number of hidden layers. 
        Each element in the iterable should contain the number of neurons to consider 
        in each list. 
    activations: Iterable[str]
        The activation functions to consider at each layer. Its dimension is n-1, as
        in the first layer (the input layer) there's no activation function being applied. 
    loss_func: str 
        Loss function to optimize. 
    weights_init: str, optional
        Weight initialization methods. 

    Returns
    -------
    dict 
        Network representation with the weights and the activation functions.
    """
    def init_weights():
        Ws = []
        weight_init_f = weights_initialization[weights_init]

        for i, layer in enumerate(architecture[1:]): 
            prev_layer = architecture[i-1]

            W_size = (layer, prev_layer)
            W = weight_init_f(W_size)
            Ws.append(W)

        return Ws

    n_samples, n_features = data.shape
    Ws = init_weights()

    for i in range(n_samples):
        sample = data[i, :]
        loss, metadata = forward_propagation(sample, Ws, activations, loss_func)
        print(loss)
