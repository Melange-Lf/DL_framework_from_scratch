from IPython.display import clear_output
import math
import numpy as np

class Model():
    def __init__(self, layers):
        self.layers = layers

        # building the layers by infering the dimensions of their matrices
        prev_dim = layers[0].out_dim  # Need to specify the inputs of the first layer
        for layer in layers:
            prev_dim = layer.build(prev_dim)
    
    def forward(self, X):

        ind = 0
        for layer in self.layers:
            X = layer.forward(X)
            ind += 1

        return X
        
    def backward(self, d_last_layer):                                                # gets the loss derivative wrt the activations of the last layer

        activation_derivatives = d_last_layer
        
        ind = 0
        for layer in self.layers:
            activation_derivatives = layer.backward(activation_derivatives, self.lr)
            ind += 1
        
    def predict(self, x):                                                                    # CURRENTLY WE ARE DOING FROM LOGITS= FALSE
        return self.forward(x)
        

    def compile(self, lr= 0.001, loss=None, metric=None):
        self.loss = loss                                                     
        self.lr = lr
        self.metric = metric
    
    def fit(self, X, Y, batch_size=64, epochs=1, val_split=0.0):
        # X: (data_points, features)
        # Y: (data_points, values)

        
        val_amount = int(X.shape[0]*val_split)
        X_val, X = X[0:val_amount, :], X[val_amount:, :]
        Y_val, Y = Y[0:val_amount, :], Y[val_amount:, :]
        batches = math.ceil(X.shape[0]/batch_size)
        shuffle_order = np.arange(Y.shape[0])
        
        for ep in epochs:
            
            np.random.shuffle(shuffle_order)
            X_ep = X[shuffle_order]
            Y_ep = Y[shuffle_order]
            
            for bat_num in range(0, batches):
                X_bat = X_ep[bat_num * batch_size : (bat_num+1) * batch_size, :]
                Y_bat = Y_ep[bat_num * batch_size : (bat_num+1) * batch_size, :]
                Y_hat = self.forward(X_bat)

                bat_loss, daL = self.loss(Y_bat, Y_hat)                                            # CURRENTLY WE ARE DOING FROM LOGITS= FALSE, AND GIVING BOTH SCALAR LOSS AND LAST LAYER ACTIVATIONS' DERIVATIVE OF LOSS
                
                self.backward(daL)

                bat_metric = self.metric(Y_bat, Y_hat)

                clear_output()
                print(f"batch_loss: {bat_loss}, batch_metric: {bat_metric}, batch: {bat_num+1}/{batches}, epoch: {ep}")                       
            clear_output()
            print(f"========================epoch_loss: {bat_loss}, epoch_metric: {bat_metric}, batch: {bat_num+1}/{batches}, epoch: {ep}")
        