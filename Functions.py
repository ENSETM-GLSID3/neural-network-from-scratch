import math
class Functions:

    @staticmethod
    def Cost( y_predicted, y ,derivate=False ):
        if derivate:
            return -y/y_predicted + (1-y)/(1-y_predicted)
        return -y*math.log(y_predicted)-(1-y)*math.log(1-y_predicted)

    def linear( x, derivate=False ):
        if derivate:
            return 1
        return x
    
    def sigmoid( x, derivate=False ):
        if derivate:
            # σ(x)(1−σ(x))
            return math.exp(-x) / (1+math.exp(-x))**2
        return 1/(1+math.exp(-x))