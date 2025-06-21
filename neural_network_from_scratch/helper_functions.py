"""
This script contains various helper functions which were created individually in respective Jupyter notebooks.
The functions in this scripts will be used to improve the modularity and reusability of the code.
This file will be updated periodically as new helper functions are created.
"""

# Imports
from typing import List, Union
import numpy as np


def neuron(input: list, weights: list, bias: float = 0.0) -> float:
    """Simulate a neuron by calculating the weighted sum of inputs plus bias.
    This function computes the dot product of the input and weights, adds a bias,
    and returns the result.

    Args:
        input (list): A list of input values.
        weights (list): A list of weights corresponding to the inputs.
        bias (float, optional): A bias term associated with the neuron. Defaults to 0.0.

    Returns:
        output (float): The output of the neuron, which is the weighted sum of inputs plus bias.

    Raises:
        ValueError: If the lengths of input and weights do not match.

    ---
    Example:
        >>> neuron([0.5, 1.5, -2.0], [0.2, -0.5, 1.0], bias=0.1)
        -2.55
    """
    # Validate input and weights lengths
    if len(input) != len(weights):
        raise ValueError("Input and weights must have the same length.")

    # Calculate the weighted sum of inputs
    output = sum(w * x for w, x in zip(weights, input)) + bias
    return output


def neuron_layer(
    inputs: Union[float, List[float]],
    weights: Union[float, List[float], List[List[float]]],
    bias: Union[float, List[float]] = 0.0,
) -> Union[float, List[float]]:
    """Simulate a layer of neurons by calculating the weighted sum of inputs plus bias for each neuron.
    This function computes the dot product of inputs and weights for each neuron in the layer,

    Args:
        inputs (Union[float, List[float]]): Input values for the layer of neurons.
        weights (Union[float, List[float], List[List[float]]]): Weights for the neurons in the layer.
            If a single float is provided, it is assumed to be the weight for a single neuron.
            If a list of floats is provided, it is assumed to be the weights for a single neuron.
            If a list of lists is provided, each inner list represents the weights for a different neuron.
        bias (Union[float, List[float]], optional): Bias term(s) for the neurons in the layer.
            If a single float is provided, it is assumed to be the bias for all neurons.
            If a list of floats is provided, each float represents the bias for a different neuron.
            Defaults to 0.0.

    Returns:
        outputs (Union[float, List[float]]): The output of the layer of neurons, which is the weighted sum of inputs plus bias for each neuron.

    Raises:
        ValueError: If the lengths of inputs and weights do not match, or if bias does not match the number of neurons.

    ---
    Example:
        >>> neuron_layer([0.5, 1.5, -2.0], [[0.2, -0.5, 1.0], [0.3, 0.1, -0.2]], bias=[0.1, -0.1])
        [-2.55, 0.60]
    """
    # Ensure inputs is a list
    if isinstance(inputs, (int, float)):
        inputs = [inputs]

    # Ensure weights is a list of lists
    if isinstance(weights, (int, float)):
        weights = [[weights]]
    if isinstance(weights, List) and not isinstance(weights[0], List):
        weights = [weights]

    # Ensure bias is a list
    if isinstance(bias, (int, float)):
        bias = [bias] * len(weights)
    elif len(bias) != len(weights):
        raise ValueError("Bias must match the number of neurons in the layer.")

    # Checking input and weights lengths
    if len(inputs) != len(weights[0]):
        raise ValueError("Input and weights must have the same length.")

    # Calculate the output for each neuron
    outputs = []
    for neuron_weights, neuron_bias in zip(weights, bias):
        output = sum(w * x for w, x in zip(neuron_weights, inputs)) + neuron_bias
        outputs.append(output)

    return outputs if len(outputs) > 1 else outputs[0]


def neuron_using_numpy(
    inputs: List[Union[int, float]],
    weights: List[Union[int, float]],
    bias: Union[int, float],
) -> float:
    """Calculates the output of a neuron using numpy.
    The function will convert inputs, weights, and bias to numpy arrays, compute the dot product of inputs and weights, add the bias, and return the result as a float.

    Args:
        inputs (List[Union[int, float]]): A list of input values to the neuron.
        weights (List[Union[int, float]]): A list of weights corresponding to the inputs.
        bias (Union[int, float]): The bias term to be added to the weighted sum.

    Returns:
        output (float): The output of the neuron as a float value.

    Raises:
        ValueError: If the lengths of inputs and weights do not match.

    ---
    Example:
        >>> neuron_using_numpy([0.5, 0.2], [0.4, 0.6], 0.1)
        0.42
    """
    if len(inputs) != len(weights):
        raise ValueError("Length of inputs and weights must match.")

    # Convert inputs, weights, and bias to numpy arrays
    inputs_array = np.array(inputs, dtype=float)
    weights_array = np.array(weights, dtype=float)
    bias_value = float(bias)

    # Calculate the dot product and add the bias
    output = np.dot(inputs_array, weights_array) + bias_value

    return output


def neuron_layer_using_numpy(
    inputs: Union[float, List[float]],
    weights: Union[float, List[float], List[List[float]]],
    biases: Union[float, List[float]] = 0.0,
) -> Union[float, List[float]]:
    """Calculates the output of a layer of neurons using numpy.
    The function will handle both single neuron and multiple neurons in a layer, allowing for flexible input and weight configurations.

    Args:
        inputs (Union[float, List[float]]): Input values to the layer, can be a single float or a list of floats.
        weights (Union[float, List[float], List[List[float]]]): Weights for the layer, can be a single float, a list of floats for a single neuron, or a 2D list for multiple neurons.
        biases (Union[float, List[float]], optional): Biases for the layer, can be a single float or a list of floats. Defaults to 0.0.

    Returns:
        Union[float, List[float]]: The output of the layer as a float if single neuron or a list of floats if multiple neurons.

    Raises:
        ValueError: If the dimensions of inputs, weights, and biases do not match appropriately.
    ---
    Example:
        >>> neuron_layer_using_numpy([0.5, 0.2], [[0.4, 0.6], [0.3, 0.8]], [0.1, 0.2])
        [0.42, 0.51]
    """

    # Convert inputs to numpy array
    if isinstance(inputs, (int, float)):
        inputs_array = np.array([float(inputs)])
    else:
        inputs_array = np.array(inputs, dtype=float)

    # Convert weights to numpy array
    if isinstance(weights, (int, float)):
        weights_array = np.array([[float(weights)]])
    elif isinstance(weights[0], (int, float)):
        weights_array = np.array([weights], dtype=float)
    else:
        weights_array = np.array(weights, dtype=float)

    # Convert biases to numpy array
    if isinstance(biases, (int, float)):
        biases_array = np.array([float(biases)])
    else:
        biases_array = np.array(biases, dtype=float)

    # Check dimensions
    if weights_array.ndim == 1:
        if inputs_array.shape[0] != weights_array.shape[0]:
            raise ValueError(
                "Length of inputs must match length of weights for a single neuron."
            )
        if (
            biases_array.shape[0] != 1
            and biases_array.shape[0] != weights_array.shape[0]
        ):
            raise ValueError(
                "Length of biases must match length of weights for a single neuron."
            )
    elif weights_array.ndim == 2:
        if inputs_array.shape[0] != weights_array.shape[1]:
            raise ValueError(
                "Length of inputs must match number of columns in weights for multiple neurons."
            )
        if biases_array.shape[0] != weights_array.shape[0]:
            raise ValueError(
                "Length of biases must match number of rows in weights for multiple neurons."
            )

    # Calculate the output
    output = np.dot(weights_array, inputs_array) + biases_array
    # Return output as float if single neuron, else return as list
    if output.ndim == 1 and output.shape[0] == 1:
        return output[0]
    else:
        return output.tolist()


def neuron_layer_with_batch_input_using_numpy(
    inputs: Union[Union[float, List[float]], List[List[float]]],
    weights: Union[float, List[float], List[List[float]]],
    biases: Union[float, List[float]] = 0.0,
) -> Union[float, np.ndarray]:
    """Calculates the output of a layer of neurons with batch input using numpy.
    The function will handle both single input and batch input scenarios, allowing for flexible configurations of inputs, weights, and biases.

    Args:
        inputs (Union[Union[float, List[float]], List[List[float]]]): List of input values to the layer, can be a single float, a list of floats for a single input, or a 2D list for batch inputs.
        weights (Union[float, List[float], List[List[float]]]): Weights for the layer, can be a single float, a list of floats for a single neuron, or a 2D list for multiple neurons.
        biases (Union[float, List[float]], optional): Biases for the layer, can be a single float or a list of floats. Defaults to 0.0.

    Returns:
        output (Union[float, np.ndarray]): The output of the layer, can be float if single input and single neuron, or a numpy array if batch input or multiple neurons.

    Raises:
        ValueError: If the dimensions of inputs, weights, and biases do not match appropriately.

    ---
    Example:
        >>> neuron_layer_with_batch_input_using_numpy([[0.5, 0.2], [0.6, 0.3]], [[0.4, 0.6], [0.3, 0.8]], [0.1, 0.2])
        array([[0.42, 0.51],
               [0.52, 0.62]])
    """

    # Convert inputs to numpy array
    if isinstance(inputs, (int, float)):
        inputs_array = np.array([[float(inputs)]])
    elif isinstance(inputs[0], (int, float)):
        inputs_array = np.array([inputs], dtype=float)
    else:
        inputs_array = np.array(inputs, dtype=float)

    # Convert weights to numpy array
    if isinstance(weights, (int, float)):
        weights_array = np.array([[float(weights)]])
    elif isinstance(weights[0], (int, float)):
        weights_array = np.array([weights], dtype=float)
    else:
        weights_array = np.array(weights, dtype=float)

    # Convert biases to numpy array
    if isinstance(biases, (int, float)):
        biases_array = np.array([float(biases)])
    else:
        biases_array = np.array(biases, dtype=float)

    # Check dimensions
    if weights_array.ndim == 1:
        if inputs_array.shape[1] != weights_array.shape[0]:
            raise ValueError(
                "Length of inputs must match length of weights for a single neuron."
            )
        if (
            biases_array.shape[0] != 1
            and biases_array.shape[0] != weights_array.shape[0]
        ):
            raise ValueError(
                "Length of biases must match length of weights for a single neuron."
            )
    elif weights_array.ndim == 2:
        if inputs_array.shape[1] != weights_array.shape[1]:
            raise ValueError(
                "Length of inputs must match number of columns in weights for multiple neurons."
            )
        if biases_array.shape[0] != weights_array.shape[0]:
            raise ValueError(
                "Length of biases must match number of rows in weights for multiple neurons."
            )

    # Calculate the output
    output = np.dot(inputs_array, weights_array.T) + biases_array

    # Return output as float if single input and single neuron, else return as numpy array
    if output.ndim == 1 and output.shape[0] == 1:
        return output[0]
    else:
        return output


def mlp_using_numpy(
    inputs: Union[float, List[float], List[List[float]]], num_layers: int, *args
) -> Union[float, List[float], List[List[float]]]:
    """
    Implements a simple Multi-Layer Perceptron (MLP) using NumPy, without activation functions.
    Supports scalar, vector, or batch input and arbitrary number of layers with custom weights/biases.

    The function expects arguments in the following sequence:
    - inputs: scalar / list / list of lists
    - num_layers: total number of layers (hidden + output)
    - *args: A flat sequence alternating weights and biases for each layer:
        (weights1, bias1, weights2, bias2, ..., weightsN, biasN)

    Each weight should be a 2D list (neurons_in_layer, input_dim),
    and each bias should be a 1D list (length = neurons_in_layer).

    Args:
        inputs (Union[float, List[float], List[List[float]]]):
            Input to the network. Can be:
                - float: scalar input
                - list of floats: single sample
                - list of list of floats: batch of samples
        num_layers (int): Number of layers in the MLP (including output layer)
        *args: Weights and biases for each layer, alternating as
               weights1, bias1, weights2, bias2, ..., weightsN, biasN.

    Returns:
        Union[float, List[float], List[List[float]]]: Output of the MLP
            - float if single scalar output
            - list of floats for single sample
            - list of list of floats for batch output

    Raises:
        ValueError: If shape mismatches or incorrect number of arguments are detected.

    Example:
        >>> mlp_using_numpy([1.0, 2.0], 2,
        ...                 [[0.5, 0.5], [0.5, 0.5]], [0.1, 0.1],
        ...                 [[0.3, 0.3], [0.3, 0.3]], [0.2, 0.2])
        [1.16, 1.16]
    """

    # Convert inputs to NumPy array and reshape if needed
    inputs = np.array(inputs, dtype=float)

    if inputs.ndim == 0:
        inputs = inputs.reshape(1, 1)  # scalar input
    elif inputs.ndim == 1:
        inputs = inputs.reshape(1, -1)  # single input vector
    elif inputs.ndim == 2:
        pass  # batch input
    else:
        raise ValueError("Input must be a scalar, 1D list, or 2D list.")

    # Validate number of layers
    if num_layers < 1:
        raise ValueError("Number of layers must be at least 1.")

    # Validate number of weight/bias arguments
    if len(args) != 2 * num_layers:
        raise ValueError(
            f"Expected {2 * num_layers} arguments for {num_layers} layers (weights and biases), but got {len(args)}."
        )

    # Forward pass through each layer
    output = inputs
    for layer in range(num_layers):
        weights = np.array(args[2 * layer], dtype=float)
        biases = np.array(args[2 * layer + 1], dtype=float)

        if weights.ndim != 2:
            raise ValueError(f"Layer {layer + 1} weights must be a 2D list or array.")
        if biases.ndim != 1:
            raise ValueError(f"Layer {layer + 1} biases must be a 1D list or array.")

        if weights.shape[1] != output.shape[1]:
            raise ValueError(
                f"Shape mismatch at layer {layer + 1}: weight expects input of shape (*, {weights.shape[1]}), got (*, {output.shape[1]})."
            )

        if biases.shape[0] != weights.shape[0]:
            raise ValueError(
                f"Shape mismatch: bias length ({biases.shape[0]}) does not match number of neurons ({weights.shape[0]}) at layer {layer + 1}."
            )

        # Forward step: output = input @ weights.T + bias
        output = output @ weights.T + biases

    # Output formatting
    if output.shape[0] == 1:
        if output.shape[1] == 1:
            return float(output[0, 0])  # scalar output
        return output.flatten().tolist()  # single sample output
    else:
        return output.tolist()  # batch output
