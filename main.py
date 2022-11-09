from ANN import *
from Functions import *



def test():
    layer = Layer(2, 2, sigmoid)
    output = layer.getOutupt( [1,0] )
    print( output )

    
    nn0 = NN([2,1], 2, [Functions.linear, Functions.linear])
    print( nn0.getOutupt([1,1]) )

    nn1 = NN([2,3], 3, [Functions.linear, Functions.linear])
    print( nn1.getOutupt([1,0,0]) )

    nn1 = NN([3,2,3], 2, [Functions.linear, Functions.linear ,Functions.linear])
    print( nn1.getOutupt([1,0]) )



def andExample():
    print()
    print("**  And Example  **")
    nn = NN([1],2,[Functions.sigmoid])
    X = [ [1,1], [0,0], [0,1], [1,0] ]
    Y = [ [1], [0], [0], [0] ]

    Y_predicted = [nn.getOutupt( x ) for x in X]
    print(Y_predicted)

    for i in range(1000):
        for x,y in zip(X,Y):
            nn.gradient_descent(x, y, 0.5)
        
    Y_predicted = [nn.getOutupt( x ) for x in X]
    print()
    print(Y)
    print(Y_predicted)

def orExample():
    print()
    print("**  Or Example  **")
    nn = NN([2,2,1], 2, [Functions.sigmoid,Functions.sigmoid, Functions.sigmoid])
    X = [ [1,1], [0,0], [0,1], [1,0] ]
    Y = [ [1], [0], [1], [1] ]

    Y_predicted = [nn.getOutupt( x ) for x in X]
    print(Y_predicted)

    for i in range(10000):
        for x,y in zip(X,Y):
            nn.gradient_descent(x, y, 0.5)
        
    Y_predicted = [nn.getOutupt( x ) for x in X]
    print(Y)
    print(Y_predicted)


def xorExample():
    print()
    print("**  Xor Example  **")
    nn = NN([2,1], 2, [Functions.sigmoid, Functions.sigmoid])
    X = [ [1,1], [0,0], [0,1], [1,0] ]
    Y = [ [0], [0], [1], [1] ]

    Y_predicted = [nn.getOutupt( x ) for x in X]
    print(Y_predicted)

    
    # nn.gradient_descent(X[0], Y[0], 0.5)
    for i in range(10000):
        for x,y in zip(X,Y):
            nn.gradient_descent(x, y, 0.00001)
        
        
    Y_predicted = [nn.getOutupt( x ) for x in X]
    print(Y)
    print(Y_predicted)

    

def test2():
    andExample()
    orExample()
    xorExample()


test2()

