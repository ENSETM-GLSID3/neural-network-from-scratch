from ANN import *

import math
def sigmoid(x):
    return 1/(math.exp(-x))
def linear(x):
    return x



def test():
    layer = Layer(2, 2, sigmoid)
    output = layer.getOutupt( [1,0] )
    print( output )

    
    nn0 = NN([2,1], 2, [linear, linear])
    print( nn0.getOutupt([1,1]) )

    nn1 = NN([2,3], 3, [linear, linear])
    print( nn1.getOutupt([1,0,0]) )

    nn1 = NN([3,2,3], 2, [linear, linear ,linear])
    print( nn1.getOutupt([1,0]) )

test()