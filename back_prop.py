import numpy as np


def relu(x): 
    return max(np.array([0]), x)


def derivative_relu(x): 
    return np.array([int(x > 0)])


def se(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Computes the Squared Error"""
    return np.sum((y_true - y_pred) ** 2)


def derivative_se(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray: 
    """Computes the partial derivatives of the SE loss function."""
    return -2 * (y_true - y_pred) 


loss_funcs = {
    "se": se,
}

loss_funcs_derivatives = {
    "se": derivative_se,        
}

activation_funcs = {
    "relu": relu,
}

activation_funcs_derivatives = {
    "relu": derivative_relu, 
}

weights_initialization = {
    "zeros": np.zeros,
    "random": np.random.rand, 
    "ones": np.ones,
}

# ----------------------------------------------------------------------
#                               Main Methods  
# ----------------------------------------------------------------------
def forward_propagation(
    X: np.ndarray, 
    y_true: np.ndarray,
    Ws: np.ndarray, 
    activations: tuple, 
    loss_f: callable
) -> tuple:
    """Compute a forward pass for a specific sample. 

    Given the weights ``Ws`, the ``activations`` and the loss function, 
    apply the forward propagation algorithm. Throughout this computation 
    the local derivatives at each neuron are also computed, being stored
    in a dictionary.

    Parameters
    ----------
    Ws: Iterable[numpy.ndarray]
        Iterable with the weight matrices to apply in the pre-activation
        computation. 
    activations: Iterable[str]
        Iterable with the name of the activation functions to use 
        for each layer.  
    loss_f: str 
        Loss function to compute in the end of the forward pass.

    Returns
    -------
    tuple[float64, dict]
        Tuple with the total loss in the end of the forward pass and the 
        metadata accumulated during the forward pass, including the local 
        derivatives computed at each neuron.
    """
    Zs = [X]
    df_dZins, dZin_dZs, dZin_dWs = [], [], []
    activation_fs = list(map(lambda name: activation_funcs[name], activations))
    derivative_fs = list(map(lambda name: activation_funcs_derivatives[name], activations))

    for i, W in enumerate(Ws): 
        activation_func = activations[i]
        print(f"\n <<< Starting FP for layer {i}>>> \n")
        print("W:", W.shape, "prev Z:", Zs[i].shape)
        # Pre-activation function
        Z_in = W @ Zs[i]
        print("After pre-activation (Z_in):", Z_in.shape)
        # Activation Function
        Z = np.apply_along_axis(activation_fs[i], 1, Z_in)
        Zs.append(Z)

        # Local derivatives
        print("Computing local derivatives...")
        df_dZin = np.apply_along_axis(derivative_fs[i], 1, Z) 
        df_dZins.append(df_dZin)

        dZin_dZ = W.T
        dZin_dZs.append(dZin_dZ)

        dZin_dW = Zs[i].T
        dZin_dWs.append(dZin_dW)

    print("y_true:", y_true, "y_pred:", Zs[-1])
    loss = loss_funcs[loss_f](y_true, Zs[-1])

    metadata = {
        # Outputs
        "Zs": Zs, 
        # Local derivatives
        "df_dZins": df_dZins,
        "dZin_dZs": dZin_dZs,
        "dZin_dWs": dZin_dWs, 
        # Loss
        "loss": loss, 
    }
    return Zs[-1], loss, metadata


def backward_propagation(y_true: np.ndarray, y_pred: np.ndarray, loss_func: str, metadata: dict) -> list: 
    """Propagates the errors backwards from the current layer `l` to the 
    previous layer `l-1`, using the chain rule to determine the contribution 
    of the components involved in computing the value of `l` to the total error. 

    Parameters
    ----------
    loss_func : str
        Represents the error contribution of the current layer `l` to the error. 
    metadata: dict
        Information collected during the forward pass, which consists in the y_pred, 
        y_true, the local partial derivatives involved in the forward pass and the 
        output of each layer.

    Returns
    -------
    list[np.array]
        The updates to apply to each Weight matrix. 
    """
    derivative_fs = loss_funcs_derivatives[loss_func]
    dloss_df = derivative_fs(y_true, y_pred)

    # Local derivatives
    df_dZins = metadata["df_dZins"] 
    dZin_dZs = metadata["dZin_dZs"]
    dZin_dWs = metadata["dZin_dWs"]

    Ws_updates = []
    n_updates = len(df_dZins) - 1

    for i in range(n_updates, -1, -1):
        print(f"\n <<< Starting BP for layer {i}>>> \n")
        df_dZin = df_dZins[i]
        dZin_dZ = dZin_dZs[i]
        dZin_dW = dZin_dWs[i]

        print(dloss_df.shape, ".", df_dZin.shape)
        dloss_dZin = np.multiply(dloss_df, df_dZin)
        print("Propagating the error to the weights at layer:", i+1)
        print(f"Local derivative\n\tdErr_dw () = dErr_dZ {dloss_dZin.shape} . Zi-1.T {dZin_dW.shape}")
        dloss_dW = dloss_dZin @ dZin_dW
        Ws_updates.append(dloss_dW)

        # Propagate the gradient for the other layer
        print("Propagating the gradient to the previous layer:", i)
        print(f"Local derivative\n\tdf_dZ(i-1) () = W.T {dZin_dZ.shape} . dloss_dZin {dloss_dZin.shape}")
        dloss_df = dZin_dZ @ dloss_dZin

    return reversed(Ws_updates)
    

def update(*args):
    pass

def train_NN(X: np.ndarray, y: np.ndarray, architecture: tuple, activations: tuple, loss_func: str, weights_init: str = "random") -> dict:
    """Trains a neural network with ``architecture`` and `activation_funcs`.

    Parameters
    ----------
    X: numpy.ndarray
        Data to fit the neural network. 

    y: numpy.ndarray
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

        for i in range(1, len(architecture)): 
            prev_layer, layer = architecture[i-1], architecture[i]

            W_size = (layer, prev_layer)
            W = weight_init_f(W_size)
            Ws.append(W)

        return Ws

    n_samples, _ = data.shape
    Ws = init_weights()
    
    # ---------------------------
    # Compute batch update
    # ---------------------------
    updates = []
    print("Processing NN with architecture:", architecture)
    print("Processing", n_samples, "samples.")
    for i in range(n_samples):
        x = X[i, :]
        x = x.reshape(len(x),1)
        
        y_true = y[i, :]
        y_true = y_true.reshape(len(y_true), 1)

        print("Sample.X:", X.shape, "Sample.y:", y_true.shape)
        y_pred, loss, metadata = forward_propagation(x, y_true, Ws, activations, loss_func)
        
        print("\n\n [END FP] \n\n")
        # Compute the backward_pass right away
        Ws_updates = backward_propagation(y_true, y_pred, loss_func, metadata) 
        Ws_sizes = map(lambda W: W.shape, Ws)
        Ws_updates_sizes = map(lambda W: W.shape, Ws_updates)

        print(list(Ws_sizes), "vs", list(Ws_updates_sizes))
        # updates += Ws_updates
    Ws = update(Ws, updates, n_samples)


data = np.array([[1, 1, 0, 1], [1, 0, 1, 2], [0, 1, 1, 3], [0, 0, 0, 4]])

n_features = 2
n_labels = 2
X = data[:, 0:n_features]
y = data[:, n_features:]

print("X:", X.shape, "y:", y.shape)
train_NN(X, y, (2, 5, 3, n_labels), ("relu", "relu", "relu", "relu"), "se", "ones")
