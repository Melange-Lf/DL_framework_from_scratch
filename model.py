from IPython.display import clear_output
import math
import numpy as np
# from torch import nn


class AvgMeter:

    def __init__(self):
        self.sum = 0
        self.num = 0
        self.avg = 0
    
    def update(self, new):
        self.num += 1
        self.sum += new
        self.avg = self.sum/self.num
    
    def __str__(self):
        return self.avg

    def rep(self):
        return self.avg


class Model():
    def __init__(self, layers, show_dims=False):
        self.layers = layers

        # building the layers by infering the dimensions of their matrices
        prev_dim = layers[0].inp_dim  # Need to specify the inputs of the first layer

        if show_dims:
            print(prev_dim)
        for layer in layers:
            prev_dim = layer.build(prev_dim)
            if show_dims:
                print(prev_dim)
    
    def forward(self, X):

        ind = 0
        for layer in self.layers:
            X = layer.forward(X)
            ind += 1

        return X
        
    def backward(self, d_last_layer):                                                # gets the loss derivative wrt the activations of the last layer

        activation_derivatives = d_last_layer
        
        ind = 0
        for layer in self.layers[::-1]:
            # print(activation_derivatives.shape, end=" :act_deri  inputs: ")
            activation_derivatives = layer.backward(activation_derivatives, self.lr)
            ind += 1
        
    def predict(self, x):                                                                    # CURRENTLY WE ARE DOING FROM LOGITS= FALSE
        return self.forward(x)
        

    def compile(self, lr= 0.001, loss=None, metric=None):
        self.loss = loss                                                     
        self.lr = lr
        self.metric = metric
    
    def fit(self, X, Y, batch_size=64, epochs=1, val_split=0.0, print_period=3):
        # X: (data_points, features)
        # Y: (data_points, values)

        
        val_amount = int(X.shape[0]*val_split)
        X_val, X = X[0:val_amount, :], X[val_amount:, :]
        Y_val, Y = Y[0:val_amount, :], Y[val_amount:, :]
        batches = math.ceil(X.shape[0]/batch_size)
        shuffle_order = np.arange(Y.shape[0])
        # sloss = nn.CrossEntropyLoss()
        
        for ep in range(epochs):
            
            np.random.shuffle(shuffle_order)
            X_ep = X[shuffle_order]
            Y_ep = Y[shuffle_order]
            ep_loss = AvgMeter()
            ep_metric = AvgMeter()
            # ep_sloss = AvgMeter()
            
            print_freq = batches // print_period
            for bat_num in range(0, batches):
                X_bat = X_ep[bat_num * batch_size : (bat_num+1) * batch_size, :]
                Y_bat = Y_ep[bat_num * batch_size : (bat_num+1) * batch_size, :]
                Y_hat = self.forward(X_bat)

                bat_loss, daL = self.loss(Y_hat, Y_bat) #Logits, Labels                                            # CURRENTLY WE ARE DOING FROM LOGITS= TRUE
                # bat_sloss = sloss(Y_hat, Y_bat)

                self.backward(daL)

                bat_metric = self.metric(Y_bat, Y_hat)

                # ep_loss.update(bat_sloss)
                ep_metric.update(bat_metric)
                ep_loss.update(bat_loss)
                
                if bat_num % print_freq == 0:
                    # print(bat_sloss)
                    print(f"\nbatch_loss: {bat_loss}, batch_metric: {bat_metric}, batch: {bat_num+1}/{batches}, epoch: {ep}")                       
                

                
            print(f"\n\n========================epoch_loss: {ep_loss.rep()}, epoch_metric: {ep_metric.rep()}, epoch: {ep}\n\n")
            # print(ep_loss.rep())
        