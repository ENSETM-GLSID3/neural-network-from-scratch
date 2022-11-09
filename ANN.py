from Functions import *
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

    def calculateDerivatives(self, inputs : list):
        self._dW = []    # matrice (repeated lines)
        self._dB = []    # 1s
        self._dZ = []    # acivation'(z)
        self._dA = []    # [...da] of previous layer

        # self._dW = inputs.copy()  # TODO: make matrix ??
        self._dW = [ inputs.copy() for _ in range( len(self.W)) ]
        self._dB = [ 1 for _ in range(len(self.W)) ]
        self._dA = [ 0 for _ in range(len(inputs)) ]

        outputs = []
        for i in range(len(self.W)):
            outputs.append(self.B[i])
            for j in range( len(self.W[0]) ):
                outputs[i] += self.W[i][j] * inputs[j]

            self._dZ.append( self.activation(outputs[i], True) )    # derive
            outputs[i] = self.activation(outputs[i])
        return outputs

    def updateParams(self, dA0, lr):            # A -> A0   ( A0 is current )
        # self._dA = [ 0 for _ in range(len(inputs)) ]
        for i in range( len(self._dW) ):
            self._dZ[i] *= dA0[i]
            self._dB[i] *= self._dZ[i]
            for j in range( len(self._dW[0]) ):
                # i fixe => neuron i ;  j fixe => neuron j of previous layer
                self._dW[i][j] *= self._dZ[i]
                self._dA[j] += self.W[i][j] * self._dZ[i]

        # UPDATING WEIGHTS
        for i in range( len(self.W) ):
            self.B[i] -= lr*self._dB[i]
            for j in range( len(self.W[0]) ):
                self.W[i][j] -= lr*self._dW[i][j]
                
        return self._dA
        

class NN:
    def __init__(self, neurons_per_layer, n_inputs, activation_functions):
        self.layers = [ ]
        for i in range( len(neurons_per_layer) ):
            self.layers.append(Layer(neurons_per_layer[i], n_inputs if i==0 else neurons_per_layer[i-1], activation_functions[i] ))
    
    def getOutupt( self, inputs:list ):
        outputs = inputs.copy()
        for l in self.layers:
            outputs = l.getOutupt(outputs)
        return outputs

    def calculateDerivatives(self, inputs):
        outputs = inputs.copy()
        for l in self.layers:
            outputs = l.calculateDerivatives(outputs)
        return outputs
    
    def gradient_descent(self, inputs, outputs, lr):
        _predicted_outs = self.calculateDerivatives(inputs)
        _dA = [ Functions.Cost(_predicted_outs[i] , outputs[i], True) for i in range(len(outputs)) ]

        for l in self.layers[::-1]:
            _dA = l.updateParams( _dA, lr )
