import gzip
import cPickle
import random

import numpy as np

def vectorize_label(x):
    result = np.zeros((10, 1))
    result[x] = 1.0
    return result

def format_dataset(data):
    images = [np.reshape(x, (784, 1)) for x in data[0]]
    labels = [vectorize_label(x) for x in data[1]]
    formatted_data = zip(images, labels)
    return formatted_data

def load_data():
    data_file = gzip.open("./data/mnist.pkl.gz", "rb")
    training_data_raw, validation_data_raw, test_data_raw = cPickle.load(data_file)
    data_file.close();

    # training_data_raw = (training_images[50,000][28 * 28 = 784], training_labels[50,000])
    # validation_data_raw = (validation_images[10,000][28 * 28 = 784], validation_labels[10,000])
    # test_data_raw = (test_images[10,000][28 * 28 = 784], test_labels[10,000])

    training_data = format_dataset(training_data_raw)
    validation_data = format_dataset(validation_data_raw)
    test_data = format_dataset(test_data_raw)

    # training_data = (training_image[28 * 28 = 784], training_label[10])[50,000]
    # validation_data = (validation_image[28 * 28 = 784], validation_label[10])[50,000]
    # test_data = (test_image[28 * 28 = 784], test_label[10])[50,000]

    return (training_data, validation_data, test_data)

def sigmoid(activation_input):
    result = (1.0 / (1.0 + np.exp(-activation_input)))
    return result

def sigmoid_derivative(activation_input):
    s = sigmoid(activation_input)
    result = s * (1.0 - s)
    return result

def derivative_cost(activation_output, y):
    result = activation_output - y
    return result

class MnistNetwork(object):
    def __init__(self, layer_sizes, weights=None, biases=None):
        self.layer_sizes = layer_sizes
        self.layer_count = len(layer_sizes)
        if (weights):
            self.weights = weights
        else:
            self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        if (biases):
            self.biases = biases
        else:
            self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]

    def feed_forward(self, data):
        input_data = data
        for weight, bias in zip(self.weights, self.biases):
            activation_input = np.dot(weight, input_data) + bias
            output_data = sigmoid(activation_input)
            input_data = output_data
        return output_data

    def backward_propagate(self, input_x, output_y):
        delta_weights = [np.zeros(weight.shape) for weight in self.weights]
        delta_biases = [np.zeros(bias.shape) for bias in self.biases]

        # feed forward while storing each layer's activation input and output
        activation_outputs = [input_x]
        activation_inputs = []
        for weight, bias in zip(self.weights, self.biases):
            activation_input = np.dot(weight, activation_outputs[-1]) + bias
            activation_inputs.append(activation_input)
            activation_outputs.append(sigmoid(activation_input))

        delta = derivative_cost(activation_outputs[-1], output_y) * sigmoid_derivative(activation_inputs[-1])
        delta_weights[-1] = np.dot(delta, activation_outputs[-2].transpose())
        delta_biases[-1] = delta

        for layer_index in xrange(2, self.layer_count):
            activation_input = activation_inputs[-layer_index]
            activation_output = activation_outputs[-layer_index - 1].transpose()
            delta = np.dot(self.weights[-layer_index + 1].transpose(), delta) * sigmoid_derivative(activation_input)
            delta_weights[-layer_index] = np.dot(delta, activation_output)
            delta_biases[-layer_index] = delta

        return (delta_weights, delta_biases)

    def update_batch(self, batch, learning_rate):
        new_weights = [np.zeros(weight.shape) for weight in self.weights]
        new_biases = [np.zeros(bias.shape) for bias in self.biases]
        for input_x, output_y in batch:
            delta_new_weights, delta_new_biases = self.backward_propagate(input_x, output_y)
            new_weights = [new_weight + delta_new_weight for new_weight, delta_new_weight in zip(new_weights, delta_new_weights)]
            new_biases = [new_bias + delta_new_bias for new_bias, delta_new_bias in zip(new_biases, delta_new_biases)]
        contribution = learning_rate / len(batch)
        self.weights = [weight - (contribution * new_weight) for weight, new_weight in zip(self.weights, new_weights)]
        self.biases = [bias - (contribution * new_bias) for bias, new_bias in zip(self.biases, new_biases)]

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def gradient_descent(self, training_data, iteration_count, batch_size, learning_rate, test_data=None):
        training_data_count = len(training_data)
        if (test_data): test_data_count = len(test_data)
        for i in xrange(iteration_count):
            random.shuffle(training_data)
            # divide training data into batches of batch_size
            batches = [training_data[j:(j + batch_size)] for j in xrange(0, training_data_count, batch_size)]
            for batch in batches:
                self.update_batch(batch, learning_rate)
            if (test_data):
                print("Iteration {0}: {1} / {2}").format(i, self.evaluate(test_data), test_data_count)
            else:
                print("Iteration {0} complete").format(i)

if __name__ == "__main__":

    training_data, validation_data, test_data = load_data()
    network = MnistNetwork([28 * 28, 30, 10])
    network.gradient_descent(training_data, 30, 10, 3.0, test_data=test_data)

    # save weights and biases to mnist_parameters.pkl and mnist_parameters.js
    '''
    cPickle.dump((network.weights, network.biases), open("mnist_parameters.pkl", "wb"))
    layer_sizes_str = ", ".join(str(e) for e in network.layer_sizes)
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    with open("mnist_parameters.js", "wb") as f:
        f.write("var layer_sizes = [" + layer_sizes_str + "];\n")
        f.write("var weights = [")
        f.write(np.array2string(network.weights[0], separator=", "))
        f.write(", ")
        f.write(np.array2string(network.weights[1], separator=", "))
        f.write("];\n")
        f.write("var biases = [")
        f.write(np.array2string(network.biases[0], separator=", "))
        f.write(", ")
        f.write(np.array2string(network.biases[1], separator=", "))
        f.write("];\n")
    '''

    # evaluate an image with the weights/biases stored in mnist_parameters.pkl
    '''
    # test_data = np.array([[<copy and paste greyscale image data here>]]).transpose()
    parameters_file = open("mnist_parameters.pkl", "rb")
    weights, biases = cPickle.load(parameters_file)
    parameters_file.close()
    network = MnistNetwork([28 * 28, 30, 10], weights=weights, biases=biases)
    result = np.argmax(network.feed_forward(test_data))
    '''

