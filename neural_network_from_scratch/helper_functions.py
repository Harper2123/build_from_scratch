"""
This script contains various helper functions which were created individually in respective Jupyter notebooks.
The functions in this scripts will be used to improve the modularity and reusability of the code.
This file will be updated periodically as new helper functions are created.
"""

# Imports
from typing import List, Union


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
