import numpy as np

class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 2
        return dx, dy

class SqrtLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x):
        self.x = np.sqrt(x)
        return self.x
    
    def backward(self, dout):
        dx = dout / (2 * np.sqrt(self.x))
        return dx

class SquareLayer:
    def __init__(self):
        self.x = None
    
    def forward(self, x):
        self.x = x**2
        return self.x
    
    def backward(self, dout):
        dx = dout * 2 * self.x
        return dx
        
class LossLayer:
    def __init__(self):
        self.l = None
    
    def forward(self, x, x_true):
        self.l = x - x_true
        loss = np.square(x - self.l)
        return loss
    
    def backward(self):
        dl = 2 * self.l
        return dl