from func import same_conv, valid_conv, transconv_valid, filter_transconv_valid, softmax, softmax_back
import numpy as np

class Conv2D(): # Currenlty for square images only
    def __init__(self, kernel=3, filters=3, padding="same", inp_dim=None):       # Use odd kernel size if same padding
        self.kernel = kernel
        self.filters = filters
        self.padding = padding
        if inp_dim:
            _ = self.build(inp_dim)
    
    def build(self, inp_dim):
        _, win_size, in_channels = inp_dim

        self.filter_arr = np.random.randn(self.filters, self.kernel, self.kernel, in_channels)*0.01

        if self.padding == "same":
            self.out_dim = (win_size, win_size, self.filters)
        elif self.padding == "valid":
            out_win_size = win_size - self.kernel + 1
            self.out_dim = (out_win_size, out_win_size, self.filters)
        else:
            print("ERROR: Only same or valid padding are allowed!")
        return self.out_dim
        
    def forward(self, X):
        if self.padding == "same":
            output = same_conv(X, self.filter_arr)
            pad = int((self.kernel - 1)/2)
        elif self.padding == "valid":
            output = valid_conv(X, self.filter_arr)
            pad = 0

        self.cache = {"layer_activ": output, "prev_layer_activ":X, "pad": pad, "bat_inp_dim": X.shape}
        return output
    
    def backward(self, activ_deri, lr):
        # getting the relevant terms for computation
        layer_activ = self.cache["layer_activ"]
        prev_layer_activ = self.cache["prev_layer_activ"]
        pad = self.cache["pad"]
        bat_size, win, _, channels = self.cache["bat_inp_dim"]
        
        if self.padding == "same":
            prev_layer_activ_padded = np.zeros((bat_size, win+2*pad, win+2*pad, channels), dtype=np.float32)
            prev_layer_activ_padded[:, pad:pad+win, pad:pad+win, channels] = prev_layer_activ
            
            filter_deri = filter_transconv_valid(activ_deri, prev_layer_activ_padded, self.kernel)
            prev_layer_activ_deri = transconv_valid(activ_deri, self.filter_arr)
            
            prev_layer_activ_deri = prev_layer_activ_deri[:, pad:pad+win, pad:pad+win, :]
            
        elif self.padding == "valid":
            filter_deri = filter_transconv_valid(activ_deri, prev_layer_activ, self.kernel)
            prev_layer_activ_deri = transconv_valid(activ_deri, self.filter_arr)

        filter_deri /= bat_size
        self.filter_arr -= filter_deri*lr
        
        return prev_layer_activ_deri
    

class Linear:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = np.random.randn(input_dim, output_dim)*0.01
        self.b = np.zeros((1, output_dim))

    def build(self, prev_outputs):
        return np.matmul(prev_outputs, self.w) + self.b

    def forward(self, X):
        output = np.matmul(X, self.w) + self.b
        self.cache = {
            'input': X,
            'output' : output
        }
        return output

    def backward(self, activation_derivatives, lr):
        X = self.cache['input']
        dW = np.matmul(X.transpose((0, 2, 1)) ,activation_derivatives)
        db = np.sum(activation_derivatives, axis=0, keepdims=True)
        dX = np.matmul(activation_derivatives, self.w.T)

        self.w-= lr * dW
        self.b-= lr * db

        return dX



class Attention():
    def __init__(self, kernel, input_dim=None):
        self.kernel = kernel
        if input_dim:
            _ = self.build(input_dim)
        
    def build(self, inp_dim):
        _, win_size, in_channels = inp_dim       # Takes in from the conv layer only and only square images`````````````````````````````````````````````````
        # The blank line is same as win_size, as we are using square pictures

        # Query, Key, Value and Output matrices
        self.Wq = np.random.randn(in_channels, self.kernel)*0.01
        self.Wk = np.random.randn(in_channels, self.kernel)*0.01
        self.Wv = np.random.randn(in_channels, self.kernel)*0.01
        self.Wo = np.random.randn(self.kernel, in_channels)*0.01
        
        self.out_dim = inp_dim
        return inp_dim
        
    def forward(self, X):
        bat_size, win_size, _, channels = X.shape
        X_flat = X.reshape((bat_size, win_size**2, channels))
        
        K = np.matmul(X_flat, self.Wk)
        Q = np.matmul(X_flat, self.Wq)
        V = np.matmul(X_flat, self.Wv)

        Q_x_K = np.matmul(Q, K.transpose((0, 2, 1)))
        Q_x_K_soft = softmax(Q_x_K)                        # ``````````````````````````````````````````````````
        Attention = np.matmul(Q_x_K_soft, V)          

        Attention_flat = np.matmul(Attention, self.Wo)

        Attention_weights = Attention_flat.reshape((bat_size, win_size, win_size, channels))

        output = X + Attention_weights

        self.cache = {"inp_dim": X.shape, "Q_x_K_soft": Q_x_K_soft, "Q_x_K":Q_x_K,
                      "V": V, "K": K, "Q": Q,
                      "Xflat": X_flat}
        return output
        
    def backward(self, activ_deri, lr):
        bat_size, win_size, _, channels = self.cache["inp_dim"]
        Attention = self.cache["Attention"]
        Q_x_K_soft = self.cache["Q_x_K_soft"]
        V = self.cache["V"]
        K = self.cache["K"]
        Q = self.cache["Q"]
        Q_x_K = self.cache["Q_x_K"]
        X_flat = self.cache["X_flat"]

        
        prev_layer_activ_deri = activ_deri
        dAttention_weights = activ_deri
        
        dAtten_flat = dAttention_weights.reshape((bat_size, win_size**2, channels))

        dWo = np.matmul(Attention.transpose((0, 2, 1)), dAtten_flat)                  
        dAttention = np.matmul(dAtten_flat, self.Wo.tranpose((1, 0)))                 

        dV = np.matmul(Q_x_K_soft.transpose((0, 2, 1)), dAttention)                  
        dQ_x_K_soft = np.matmul(dAttention, V.transpose((0, 2, 1)))                 

        dQ_x_K = softmax_back(dQ_x_K_soft)             # -------------------------------------------                                 
        dQ = np.matmul(dQ_x_K, K)                                                
        dKt = np.matmul(Q.transpose((0, 2, 1)), dQ_x_K)                                  
        dK = dKt.tranpose((0, 2, 1))

        dX_flat = np.matmul(dQ, self.Wq.tranpose((1, 0))) + np.matmul(dK, self.Wk.tranpose((1, 0))) + np.matmul(dV, self.Wv.tranpose((1, 0)))
        dWq = np.matmul(X_flat.tranpose((0, 2, 1)), dQ)
        dWk = np.matmul(X_flat.tranpose((0, 2, 1)), dK)
        dWv = np.matmul(X_flat.tranpose((0, 2, 1)), dK)

        prev_layer_activ_deri += dX_flat.reshape((bat_size, win_size, win_size, channels))

        self.Wq -= lr*dWq.mean(axis=0)
        self.Wk -= lr*dWk.mean(axis=0)
        self.Wv -= lr*dWv.mean(axis=0)
        self.Wo -= lr*dWo.mean(axis=0)
        
        return prev_layer_activ_deri
    

    
class Sigmoid:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def build(self, prev_outputs):
        return prev_outputs

    def forward(self, X):
        output = 1/(1 + np.exp(-X))
        self.cache = {
            'input': X,
            'output' : output
        }
        return output

    def backward(self, activation_derivatives, lr):
        sigmoid_output = self.cache['output']
        return activation_derivatives*sigmoid_output*(1 - sigmoid_output)
