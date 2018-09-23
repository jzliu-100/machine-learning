# Source: https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1

from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output

            print("The error between the output and training_output is:")
            print(error)

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

            print("The adjustment for the weights is:")
            print(adjustment)
            print("The new and adjusted weights are:")
            print(self.synaptic_weights)
            print("")

    # The neural network thinks.
    def think(self, inputs):
        print("Thinking..................")
        print("inputs are:")
        print(inputs)
        print("synaptic weights are:")
        print(self.synaptic_weights)
        print("The dot product of the inputs and synaptic weights is:")
        dot_product=dot(inputs, self.synaptic_weights)
        print(dot_product)
        print("The output is:")
        sigmoid_output=self.__sigmoid(dot_product)
        print(sigmoid_output)
        
        # Pass inputs through our neural network (our single neuron).
        return sigmoid_output

# if a module is being run directly, then __name__ instead is set to the string "__main__". 
# If your script is being imported into another module, its various function and class definitions 
# will be imported and its top-level code will be executed, but the code in the then-body of the 
# if clause above won't get run as the condition is not met.
# Thus, you can test whether your script is being run directly or being imported by something else by testing
if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # try another training samples and outputs
    # training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0],[1,0,0],[1,1,1],[0,0,0]])
    # training_set_outputs = array([[0, 1, 1, 1,1,0,0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("New synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    # Test the neural network with a new situation.
    print("Considering new situation [1, 0, 0] -> ?: ")
    # print(neural_network.think(array([1, 0, 0])))
    print(neural_network.think(array([1, 1, 0])))