import numpy as np


def forward_propagation(l: np.ndarray, f: str, W: np.ndarray) -> np.ndarray:
    """Given the output values of a preceeding layer ``l`` compute the 
    output of the proceeding layer,  using the specified weights
    ``W`` and the activation function ``f``.

    Parameters
    ----------
    l: numpy.ndarray
        Vector with the output values of the preceeding layer. 
    f: str
        Name of the activation function to use for the computation 
        of the output value of the layer. 
    W: numpy.ndarray
        Matrix with the vectors to be used for the computation of
        the output layer. It should be of the dimension mxn where 
        m corresponds to the number of neurons in the proceeding 
        layer and ``n```corresponds to the number of neurons in 
        ``l``.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        Tuple with the vector representing the output values of the 
        next layer and vector with the local derivatives that result 
        from the computation done in this step. 
    """
    pass

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
    

def train_NN(architecture: tuple, activation_funcs: tuple, loss_func: str, weights_init: str = "random") -> dict:
    """Trains a neural network with ``architecture`` and `activation_funcs`.

    Parameters
    ----------
    architecture: Iterable[int]
        An iterable whose dimension is n+2, where n is the number of hidden layers. 
        Each element in the iterable should contain the number of neurons to consider 
        in each list. 
    activation_funcs: Iterable[str]
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
    pass
