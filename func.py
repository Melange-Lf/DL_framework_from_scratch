import numpy as np
import scipy as sp

# NOTE: Debug 'PRINT( '


def softmax(inp, axis = -1):
    
    inp -= np.max(inp, axis=axis, keepdims=True) # stability
    exp_x = np.exp(inp)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def cross_entropy_loss(logits, labels):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits/np.sum(exp_logits, axis=-1, keepdims=True)
    
    log_probs = -np.log(probs + 1e-9)
    loss = np.mean(np.sum(log_probs*labels, axis=-1, dtype=np.float64))
    d_logits = (probs - labels)/logits.shape[0]
    
    # print(loss)
    return loss, d_logits

def accuracy(logits, labels):
    true_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)
    return np.mean(true_classes == pred_classes)


# ================================================ INTERNAL FUNCTIONS BELOW





def same_conv(target_arr, filt):
    # implemented batches
    # print( "same conv start")
    bat_size, win_size, _, channels = target_arr.shape
    filt_num, kernel, _, _ = filt.shape
    out_win_size = win_size
    pad = int((kernel-1)/2)

    intermediate = np.zeros((bat_size, win_size+2*pad, win_size+2*pad, channels), dtype=np.float32)
    intermediate[:, pad:-pad, pad:-pad, :] = target_arr
    # print( f"same conv variable defined, running for {bat_size} batches, {filt_num} filters")
    print_freq = bat_size // 2
    
    output = np.zeros((bat_size, out_win_size, out_win_size, filt_num), dtype=np.float32)
    for data in range(bat_size):
        # if data % print_freq == 0:
            # pass
            # print( f"current bat: {data}")
        for fltr in range(filt_num):
            
            
            output[data, :, :, fltr] = sp.signal.correlate(intermediate[data, :, :, :], filt[fltr, :, :, :],
                                                           mode = 'valid').reshape((out_win_size, out_win_size))
    
    return output

def valid_conv(target_arr, filt):
    # implemented batches
    # print( "valid conv start")
    bat_size, win_size, _, channels = target_arr.shape
    filt_num, kernel, _, _ = filt.shape
    out_win_size = win_size - kernel + 1
    
    # print( f"valid conv variable defined, running for {bat_size} batches, {filt_num} filters")
    print_freq = bat_size // 2

    output = np.zeros((bat_size, out_win_size, out_win_size, filt_num), dtype=np.float32)
    for data in range(bat_size):
        # if data % print_freq == 0:
            # pass
            # print( f"current bat: {data}")
        for fltr in range(filt_num):
            output[data, :, :, fltr] = sp.signal.correlate(target_arr[data, :, :, :], filt[fltr, :, :, :],
                                                           mode = 'valid').reshape((out_win_size, out_win_size))
    
    return output

# For same version just remove the outer layer from the output, which is supposed to be the padded part
# Could alternatively write function decorator
def transconv_valid(input_arr, filters):
    # Function used to get the previous layer activations derivatives
    bat_size, _, inp_win, num_filt = input_arr.shape
    num_filt, _, kernel, channels = filters.shape

    out_win = inp_win + kernel - 1
    output = np.zeros((bat_size, out_win, out_win, channels), dtype = np.float32)

    for data in range(bat_size):
        for filt in range(num_filt):
            for row in range(inp_win):
                for col in range(inp_win):
                    output[data, row:kernel+row, col:kernel+col, :] += filters[filt, :, :, :]*input_arr[data, row, col, filt]
    return output


# For the same version just pass in a zero padded layer_activation
# could alternatively write decorator in future
def filter_transconv_valid(input_arr, layer_activations, kernel):
    # used to get the filter's derivatives
    bat_size, _, inp_win, num_filt = input_arr.shape
    bat_size, lay_win, _, channels = layer_activations.shape

    output = np.zeros((num_filt, kernel, kernel, channels), dtype = np.float32)

    # print(output.shape, layer_activations.shape,input_arr.shape)
    for data in range(bat_size):
        for filt in range(num_filt):
            for row in range(inp_win):
                for col in range(inp_win):
                    output[filt, :, :, :] += layer_activations[data, row:kernel+row, col:kernel+col, :]*input_arr[data, row, col, filt]
    return output

# Need to look into this
# same_vec = np.frompyfunc(same_conv, 2, 1)
# valid_vec = np.frompyfunc(valid_conv, 2, 1)

def softmax_back(grad_output, softmax_output):

    # dy_j/dx_i = -yj * yi
    # dy_i/dy_i = yi(1-yi)
    # dL/dx_i   =  SUM[  dL/dy_k  * dy_k/dx_i   ]
    
    # used expression below:   dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
    sum_term = np.sum(grad_output * softmax_output, axis=-1, keepdims=True)
    return softmax_output * (grad_output - sum_term)

