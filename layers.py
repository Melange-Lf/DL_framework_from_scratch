from func import same_conv, valid_conv, transconv_valid, filter_transconv_valid, softmax, softmax_back
import numpy as np

# NOTE: Debug 'print( '


class legacy_Conv2D(): # Currenlty for square images only
    def __init__(self, kernel=3, filters=3, padding="same", inp_dim=None):       # Note: Use odd kernel size if same padding
        self.kernel = kernel
        self.filters = filters
        self.padding = padding.lower()  
        self.inp_dim = inp_dim
        if inp_dim:
            _ = self.build(inp_dim)
    
    def build(self, inp_dim):
        if len(inp_dim) != 3:
            raise ValueError(f"Expected input dimension to be (win_size, win_size, channels), got {inp_dim}")
            
        _, win_size, in_channels = inp_dim

        # Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (self.kernel * self.kernel * in_channels))
        self.filter_arr = np.random.randn(self.filters, self.kernel, self.kernel, in_channels) * scale

        if self.padding == "same":
            out_dim = (win_size, win_size, self.filters)
        elif self.padding == "valid":
            out_win_size = win_size - self.kernel + 1
            out_dim = (out_win_size, out_win_size, self.filters)
        else:
            raise ValueError(f"Invalid padding: {self.padding}. Only 'same' or 'valid' are allowed, (no case sensitivity)")

        self.out_dim = out_dim
        return self.out_dim
        
    def forward(self, X):
        # print( "control to conv subprocesses")
        if self.padding == "same":
            output = same_conv(X, self.filter_arr)
            pad = int((self.kernel - 1)/2)
        elif self.padding == "valid":
            output = valid_conv(X, self.filter_arr)
            pad = 0
        else:
            raise ValueError(f"Invalid padding: {self.padding}. Only 'same' or 'valid' are allowed.")

        self.cache = {"layer_activ": output, "prev_layer_activ": X, "pad": pad, "bat_inp_dim": X.shape}
        return output
    
    def backward(self, activ_deri, lr):
        # getting the relevant terms for computation
        layer_activ = self.cache["layer_activ"]
        prev_layer_activ = self.cache["prev_layer_activ"]
        pad = self.cache["pad"]
        bat_size, win, _, channels = self.cache["bat_inp_dim"]
        
        if self.padding == "same":
            prev_layer_activ_padded = np.zeros((bat_size, win+2*pad, win+2*pad, channels), dtype=np.float32)
            prev_layer_activ_padded[:, pad:pad+win, pad:pad+win, :] = prev_layer_activ
            
            filter_deri = filter_transconv_valid(activ_deri, prev_layer_activ_padded, self.kernel)
            prev_layer_activ_deri = transconv_valid(activ_deri, self.filter_arr)
            
            prev_layer_activ_deri = prev_layer_activ_deri[:, pad:pad+win, pad:pad+win, :]
            
        elif self.padding == "valid":
            filter_deri = filter_transconv_valid(activ_deri, prev_layer_activ, self.kernel)
            prev_layer_activ_deri = transconv_valid(activ_deri, self.filter_arr)

        filter_deri /= bat_size
        self.filter_arr -= filter_deri*lr
        
        return prev_layer_activ_deri
    import numpy as np


# Implementation of the article https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/
class Conv2D:
    def __init__(self, kernel_size, filters, padding='valid', inp_dim=None): # Channels first
        if isinstance(kernel_size, int):
            self.kernel_h = self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size
        
        self.filters = filters
        self.padding = padding
        self.inp_dim = inp_dim
        
        if inp_dim is not None:
            self.build(inp_dim)
    
    def build(self, input_shape):
        self.input_channels, self.input_h, self.input_w = input_shape
        
        fan_in = self.input_channels * self.kernel_h * self.kernel_w
        fan_out = self.filters * self.kernel_h * self.kernel_w
        limit = np.sqrt(6 / (fan_in + fan_out))
        
        self.w = np.random.uniform(-limit, limit, 
                                 (self.filters, self.input_channels, self.kernel_h, self.kernel_w))
        self.b = np.zeros((self.filters,))
        
        if self.padding == 'same':
            self.pad_h = (self.kernel_h - 1) // 2
            self.pad_w = (self.kernel_w - 1) // 2
            self.out_h = self.input_h
            self.out_w = self.input_w
        else:  # 'valid'
            self.pad_h = self.pad_w = 0
            self.out_h = self.input_h - self.kernel_h + 1
            self.out_w = self.input_w - self.kernel_w + 1
        
        return (self.filters, self.out_h, self.out_w)
    
    def _pad_input(self, X):
        if self.pad_h == 0 and self.pad_w == 0:
            return X
        return np.pad(X, ((0, 0), (0, 0), (self.pad_h, self.pad_h), (self.pad_w, self.pad_w)), 
                     mode='constant', constant_values=0)
    
    def _im2col(self, X):
        batch, channels, h, w = X.shape
        
        cols = np.zeros((batch * self.out_h * self.out_w, channels * self.kernel_h * self.kernel_w))
        
        for y in range(self.out_h):
            y_max = y + self.kernel_h
            for x in range(self.out_w):
                x_max = x + self.kernel_w

                patches = X[:, :, y:y_max, x:x_max]  # (batch, channels, kh, kw)
                patches_flat = patches.reshape(batch, -1)  # (batch, channels*kh*kw)
                
                idx = y * self.out_w + x
                cols[idx::self.out_h*self.out_w, :] = patches_flat # steps over every batch
        
        return cols
    
    
    
    def forward(self, X):
        # X == (batch, channels, height, width)
        
        batch = X.shape[0]
        
        X_padded = self._pad_input(X)
        
        # Preparing patch vectors
        X_col = self._im2col(X_padded)  # (batch*out_h*out_w, channels*kh*kw) (order important)
        
        W_col = self.w.reshape(self.filters, -1)  # (filters, channels*kh*kw) (order important)
        
        output = np.dot(X_col, W_col.T)  # (batch*out_h*out_w, filters)
        
        output += self.b
        
        output = output.reshape(batch, self.out_h, self.out_w, self.filters)
        output = output.transpose(0, 3, 1, 2)  # (batch, filters, out_h, out_w)
        
        self.cache = {
            'input': X,
            'input_padded': X_padded,
            'input_col': X_col,
            'output': output
        }
        
        return output
    
    
    def _col2im(self, cols, input_shape):
        batch, channels, h, w = input_shape
        X = np.zeros(input_shape)
        
        for y in range(self.out_h):
            y_max = y + self.kernel_h
            for x in range(self.out_w):
                x_max = x + self.kernel_w
                idx = y * self.out_w + x
                
                grad_patch = cols[idx::self.out_h*self.out_w, :]  # (batch, channels*kh*kw)
                grad_patch = grad_patch.reshape(batch, channels, self.kernel_h, self.kernel_w)
                
                X[:, :, y:y_max, x:x_max] += grad_patch
        
        return X

    def backward(self, activation_derivatives, lr):
        batch = activation_derivatives.shape[0]
        
        dout = activation_derivatives.transpose(0, 2, 3, 1)  # (batch, out_h, out_w, filters)
        dout_col = dout.reshape(-1, self.filters)  # (batch*out_h*out_w, filters)
        
        # Gradient weights
        X_col = self.cache['input_col'] # (batch*out_h*out_w, channels*kh*kw)
        dW = np.dot(dout_col.T, X_col)  # (filters, channels*kh*kw) (note the ordering)
        dW = dW.reshape(self.w.shape)  # (filters, channels, kh, kw)
        
        # Gradient bias
        db = np.sum(dout_col, axis=0)  # (filters,)
        
        # Gradient input
        W_col = self.w.reshape(self.filters, -1)  # (filters, channels*kh*kw)
        dX_col = np.dot(dout_col, W_col)  # (batch*out_h*out_w, channels*kh*kw)
        
        # Assigning back grads from each output pixel to its patch
        dX_padded = self._col2im(dX_col, self.cache['input_padded'].shape) # (batch, channels, height, width)
        
        # Rem padding
        if self.pad_h > 0 or self.pad_w > 0:
            dX = dX_padded[:, :, self.pad_h:self.pad_h+self.input_h, self.pad_w:self.pad_w+self.input_w]
        else:
            dX = dX_padded
        
        self.w -= lr * dW / batch
        self.b -= lr * db / batch
        
        return dX
    

class Linear:
    def __init__(self, inp_dim, out_dim):
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.w = np.random.randn(inp_dim, out_dim)*0.1
        self.b = np.zeros((1, out_dim))

    def build(self, inp_dim):
        assert inp_dim == self.inp_dim, "previous layer's output shape does not match input shape"
        return self.out_dim

    def forward(self, X):
        
        output = np.matmul(X, self.w) + self.b
        self.cache = {
            'input': X,
            'output' : output
        }
        # print(X.shape, output.shape)
        return output

    def backward(self, activation_derivatives, lr):
        X = self.cache['input']
        # print(X.shape)
        dW = np.matmul(X.T ,activation_derivatives)
        db = np.sum(activation_derivatives, axis=0, keepdims=True)
        dX = np.matmul(activation_derivatives, self.w.T)

        batch = X.shape[0]
        self.w-= lr * dW/batch
        self.b-= lr * db/batch

        return dX



class Attention():
    def __init__(self, kernel, inp_dim=None):
        self.kernel = kernel
        if inp_dim:
            _ = self.build(inp_dim)
        
    def build(self, inp_dim):
        in_channels, height, width = inp_dim
        
        assert height == width, "Attention currently implemented for Square inputs only"
        # We are using square pictures

        # Query, Key, Value and Output matrices
        self.Wq = np.random.randn(in_channels, self.kernel)*0.01
        self.Wk = np.random.randn(in_channels, self.kernel)*0.01
        self.Wv = np.random.randn(in_channels, self.kernel)*0.01
        self.Wo = np.random.randn(self.kernel, in_channels)*0.01
        
        self.out_dim = inp_dim
        return inp_dim
        
    def forward(self, X):
        bat_size, channels, win_size, _ = X.shape
        
        X_flat = X.transpose(0, 2, 3, 1).reshape((bat_size, win_size**2, channels))
        
        K = np.matmul(X_flat, self.Wk)
        Q = np.matmul(X_flat, self.Wq)
        V = np.matmul(X_flat, self.Wv)

        Q_x_K = np.matmul(Q, K.transpose((0, 2, 1)))/np.sqrt(self.kernel) # ----------------------------
        
        Q_x_K_soft = softmax(Q_x_K)                        # ``````````````````````````````````````````````````
        Attention = np.matmul(Q_x_K_soft, V)          

        Attention_flat = np.matmul(Attention, self.Wo)

        Attention_weights = Attention_flat.reshape((bat_size, win_size, win_size, channels)).transpose(0, 3, 1, 2)

        output = X + Attention_weights

        self.cache = {"inp_dim": X.shape,
                      "X_flat": X_flat,
                      "V": V, 
                      "K": K, 
                      "Q": Q,
                      "Q_x_K":Q_x_K,
                      "Q_x_K_soft": Q_x_K_soft,
                      "Attention" : Attention
        }
        return output
        
    def backward(self, activ_deri, lr):
        bat_size, channels, win_size, _ = self.cache["inp_dim"]
        Attention = self.cache["Attention"]
        Q_x_K_soft = self.cache["Q_x_K_soft"]
        V = self.cache["V"]
        K = self.cache["K"]
        Q = self.cache["Q"]
        X_flat = self.cache["X_flat"]

        
        prev_layer_activ_deri = activ_deri
        dAttention_weights = activ_deri
        
        dAtten_flat = dAttention_weights.transpose(0, 2, 3, 1).reshape((bat_size, win_size**2, channels))

        dWo = np.matmul(Attention.transpose((0, 2, 1)), dAtten_flat)                  
        dAttention = np.matmul(dAtten_flat, self.Wo.T)                 

        dV = np.matmul(Q_x_K_soft.transpose((0, 2, 1)), dAttention)                  
        dQ_x_K_soft = np.matmul(dAttention, V.transpose((0, 2, 1)))                 

        dQ_x_K = softmax_back(dQ_x_K_soft, Q_x_K_soft) / np.sqrt(self.kernel)             # -------------------------------------------                                 
        dQ = np.matmul(dQ_x_K, K)                                                
        dK = np.matmul(dQ_x_K.transpose(0, 2, 1), Q)

        dX_flat = (np.matmul(dQ, self.Wq.T) + 
                   np.matmul(dK, self.Wk.T) + 
                   np.matmul(dV, self.Wv.T)  )
        dWq = np.matmul(X_flat.transpose((0, 2, 1)), dQ)
        dWk = np.matmul(X_flat.transpose((0, 2, 1)), dK)
        dWv = np.matmul(X_flat.transpose((0, 2, 1)), dK)

        prev_layer_activ_deri += dX_flat.reshape((bat_size, win_size, win_size, channels)).transpose(0, 3, 1, 2)

        self.Wq -= lr*dWq.mean(axis=0)
        self.Wk -= lr*dWk.mean(axis=0)
        self.Wv -= lr*dWv.mean(axis=0)
        self.Wo -= lr*dWo.mean(axis=0)
        
        return prev_layer_activ_deri
    
class Flatten:
    def __init__(self):
        pass
    
    def build(self, prev_outputs):
        self.input_shape = prev_outputs
        flattened_size = np.prod(prev_outputs)
        return int(flattened_size)
    
    def forward(self, X):
        self.cache = {'input_shape': X.shape}
        
        if len(X.shape) == 2:
            return X
        
        batch_size = X.shape[0]
        flattened = X.reshape(batch_size, -1)
        return flattened
    
    def backward(self, activation_derivatives, lr=None):
        original_shape = self.cache['input_shape']
        
        if len(original_shape) == 2:
            return activation_derivatives
        
        return activation_derivatives.reshape(original_shape)


class LazyLinear:
    def __init__(self, out_dim):
        self.out_dim = out_dim
        self.inp_dim = None
        self.w = None
        self.b = None
        self.built = False
    
    def build(self, prev_outputs):
        self.inp_dim = prev_outputs
        
        self.w = np.random.randn(self.inp_dim, self.out_dim) * 0.1
        self.b = np.zeros((1, self.out_dim))
        
        return self.out_dim
    
    def forward(self, X):
        
        if len(X.shape) > 2:
            batch_size = X.shape[0]
            X = X.reshape(batch_size, -1)
        
        output = np.matmul(X, self.w) + self.b
        self.cache = {
            'input': X,
            'output': output
        }
        return output
    
    def backward(self, activation_derivatives, lr):
        
        X = self.cache['input']
        dW = np.matmul(X.T, activation_derivatives)
        db = np.sum(activation_derivatives, axis=0, keepdims=True)
        dX = np.matmul(activation_derivatives, self.w.T)
        
        batch = X.shape[0]
        self.w -= lr * dW / batch
        self.b -= lr * db / batch
        
        return dX



    
class Sigmoid:
    def __init__(self, inp_dim, out_dim):
        self.inp_dim = inp_dim
        self.out_dim = out_dim

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


class ReLU:
    def __init__(self, inp_dim=None, out_dim=None):
        self.inp_dim = inp_dim
        self.out_dim = out_dim

    def build(self, prev_outputs):
        return prev_outputs

    def forward(self, X):
        output = np.maximum(0, X)
        self.cache = { 'input': X }
        return output

    def backward(self, activation_derivatives, lr):
        X = self.cache['input']
        grad_mask = (X > 0).astype(X.dtype)
        return activation_derivatives * grad_mask