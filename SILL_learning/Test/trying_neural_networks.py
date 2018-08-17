import numpy as np
# My first attempt at building neural nets from scratch.
# HELP FROM: https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-
# network-in-9-lines-of-python-code-cc8f23647ca1
# AND: https://medium.com/technology-invention-and-more/how-to-build-a-multi-layered-neural-
# network-in-python-53ec3d1d326a
# AND: http://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

class NeuronLayer():
    """

    """

    def __init__(self, layer_weights):
        """
        :param weights: an n by m numpy array, where n is the number of
        inputs per neuron and m is the number of neurons
        """
        self.layer_weights = layer_weights


class NeuralNetwork():
    """
    A single neuron solves a linear problem.
    in->node->out
    Or, try the 2 layer neural net.
    """

    def __init__(self, initial_weights, layer1, layer2):
        """
        Gives the initial weights for the simple case
        Gives the weights on the two layers for the 2 layer case
        :param initial_weights: 
        :param layer1: 
        :param layer2: 
        """
        self.weights = initial_weights
        self.layer1 = layer1
        self.layer2 = layer2

    def _sigmoid(self, x):
        """
        Returns the value of the sigmoid function, this is the model function
        that the neural network uses to approximate the output.
        :param x: 
        :return: 
        """
        return (1 + np.exp(-x))**(-1)

    def _sigmoid_grad(self, x):
        """
        Returns the gradient of the sigmoid function
        :param x: 
        :return: 
        """
        return x * (1 - x)

    def simpletrain(self, train_inputs, train_outputs, num_iters):
        """
        Trains the weights of the simple nn through an iterative process
        :param train_inputs: 
        :param train_outputs: 
        :param num_iters: 
        :return: 
        """
        # for each input, predict the output
        # then calculate the error
        # and adjust the weights
        for i in range(num_iters):
            inpt = train_inputs[i]
            output = self.simplepredict(inpt)
            error = np.subtract(output, train_outputs[i])
            adjust = inpt.dot(error * self._sigmoid_grad(output))
            self.weights += adjust

    def simplepredict(self, inputs):
        """
        The simple nn gives a prediction.
        :param inputs: 
        :return: 
        """
        return self._sigmoid(inputs.dot(self.weights))

    def twotrain(self, train_inputs, train_outputs, num_iters):
        """
        The two layer nn trains iteratively
        :param train_inputs: 
        :param train_outputs: 
        :param num_iters: 
        :return: 
        """
        # for each input, predict the output
        # then calculate the error for layer 2 and layer 1
        # and adjust the weights on both layers
        for i in range(num_iters):
            print(i)
            inpt = train_inputs[i]
            #print("layer2 weights", self.layer2.layer_weights)
            #print("layer1 weights", self.layer1.layer_weights)
            output1, output2 = self.twopredict(inpt)
            layer2_error = (train_outputs[i] - output2) * self._sigmoid_grad(output2)
            #print("layer1 weights", self.layer1.layer_weights)
            #print("output1", output1)
            #print("layer2 error", layer2_error)
            layer1_error = layer2_error * self.layer1.layer_weights.dot(self._sigmoid_grad(output1))
            #print("input", inpt)
            #print("later1_error", layer1_error)
            adjustment1 = inpt.dot(layer1_error)
            adjustment2 = output1 * layer2_error  #TODO: Fix the error and adjustments
            print("adjustment 2", adjustment2)
            print("layer2 weights", self.layer2.layer_weights)
            self.layer1.layer_weights += adjustment1
            self.layer2.layer_weights += [[i] for i in adjustment2]

    def twopredict(self, inputs):
        """
        The 2 layer nn gives a prediction, this is forward propagation.
        :param inputs: 
        :return: 
        """
        layer1_output = self._sigmoid(inputs.dot(self.layer1.layer_weights))
        layer2_output = self._sigmoid(layer1_output.dot(self.layer2.layer_weights))
        return layer1_output, layer2_output


test_inputs = np.array([[0, 0, 1],
                        [1, 1, 1],
                        [1, 0, 1],
                        [0, 1, 1]])
test_outputs = np.array([0, 1, 1, 0])
two_test = np.array([[0, 0, 1],
                     [0, 1, 1],
                     [1, 0, 1],
                     [0, 1, 0],
                     [1, 0, 0],
                     [1, 1, 1],
                     [0, 0, 0]])
two_outputs = np.array([0, 1, 1, 1, 1, 0, 0])

nn = NeuralNetwork(np.array([1, 1., 1]), NeuronLayer(np.ones([3, 4])), NeuronLayer(np.ones([4, 1])))
nn.simpletrain(test_inputs, test_outputs, 4)
nn.simplepredict(np.array([1, 0, 0]))
nn.twotrain(two_test, two_outputs, 7)
print(nn.twopredict(np.array([1, 1, 0])))
# It runs but it stinks!
