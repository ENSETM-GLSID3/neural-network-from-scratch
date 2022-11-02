class Layer:
    def __init__(self, n_neurons, n_inputs, activation):
        self.B = [ 0 for i in range(n_neurons) ]
        self.W = []
        self.activation = activation
        for i in range(n_neurons):
            self.W.append([ 1 for i in range(n_inputs) ])

    def getOutupt(self, inputs):
        outputs = []
        for i in range(len(self.W)):
            outputs.append(self.B[i])
            for j in range( len(self.W[0]) ):
                outputs[i] += self.W[i][j] * inputs[j]

            outputs[i] = self.activation(outputs[i])
        return outputs


class NN:
    def __init__(self, neurons_per_layer, n_inputs, activation_functions):
        self.layers = [ ]
        for i in range( len(neurons_per_layer) ):
            self.layers.append(Layer(neurons_per_layer[i], n_inputs if i==0 else neurons_per_layer[i-1], activation_functions[i] ))
    
    def getOutupt( self, inputs:list() ):
        outputs = inputs.copy()
        for l in self.layers:
            outputs = l.getOutupt(outputs)
        return outputs
