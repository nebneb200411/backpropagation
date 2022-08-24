import numpy as np
from layers import *

class NetWork:
    def __init__(self, training_num, learning_rate=0.01, z_true=0):
        self.z_true = z_true
        self.add_layer = AddLayer()
        self.square_layer_x1 = SquareLayer()
        self.square_layer_y1 = SquareLayer()
        self.sqrt_layer = SqrtLayer()
        self.loss_layer = LossLayer()
        self.learning_rate = learning_rate
        self.training_num = training_num
        self.x_record = []
        self.y_record = []
    
    def main(self, initial_x, initial_y):
        self.x_record.append(initial_x)
        self.y_record.append(initial_y)

        x = initial_x
        y = initial_y
        
        for epoch in range(self.training_num):
            x, y = self.training(x, y)
            self.x_record.append(x)
            self.y_record.append(y)
        
        print('training fin')
        return self.x_record, self.y_record
            
    
    def training(self, x, y):
        loss = self.forward(x, y)
        dx1dx, dy1dy = self.backward(loss)

        # update
        x = x - self.learning_rate * x
        y = y - self.learning_rate * y

        return x, y
    
    def forward(self, x_input, y_input):
        # forward
        # x1 = x**2
        # y1 = y**2
        x1 = self.square_layer_x1.forward(x_input)
        y1 = self.square_layer_y1.forward(y_input)

        z1 = self.add_layer.forward(x1, y1)

        z2 = self.sqrt_layer.forward(z1)

        loss = self.loss_layer.forward(z2, self.z_true)

        return loss
    
    def backward(self, loss):
        dldz2 = self.loss_layer.backward()

        dz2dz1 = self.sqrt_layer.backward(dldz2)

        dz1dx1,dz1dy1 = self.add_layer.backward(dz2dz1)

        dx1dx = self.square_layer_x1.backward(dz1dx1)
        dy1dy = self.square_layer_x1.backward(dz1dy1)

        return dx1dx, dy1dy
        


        


