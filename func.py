import numpy as np
import scipy as sp

def same_conv(target_arr, filt):
    # implemented batches
    bat_size, win_size, _, channels = target_arr.shape
    filt_num, kernel, _, _ = filt.shape
    out_win_size = win_size
    pad = int((kernel-1)/2)

    intermediate = np.zeros((bat_size, win_size+2*pad, win_size+2*pad, channels), dtype=np.float32)
    intermediate[:, pad:-pad, pad:-pad, :] = target_arr
    
    output = np.zeros((bat_size, out_win_size, out_win_size, filt_num), dtype=np.float32)
    for data in range(bat_size):
        for fltr in range(filt_num):
            
            output[data, :, :, fltr] = sp.signal.correlate(intermediate[data, :, :, :], filt[fltr, :, :, :],
                                                           mode = 'valid').reshape((out_win_size, out_win_size))
    
    return output

def valid_conv(target_arr, filt):
    # implemented batches
    bat_size, win_size, _, channels = target_arr.shape
    filt_num, kernel, _, _ = filt.shape
    out_win_size = win_size - kernel + 1
    
    output = np.zeros((bat_size, out_win_size, out_win_size, filt_num), dtype=np.float32)
    for data in range(bat_size):
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

    print(output.shape, layer_activations.shape,input_arr.shape)
    for data in range(bat_size):
        for filt in range(num_filt):
            for row in range(inp_win):
                for col in range(inp_win):
                    output[filt, :, :, :] += layer_activations[data, row:kernel+row, col:kernel+col, :]*input_arr[data, row, col, filt]
    return output

# Need to look into this
# same_vec = np.frompyfunc(same_conv, 2, 1)
# valid_vec = np.frompyfunc(valid_conv, 2, 1)


def softmax_back(deri): # current placeholder
    pass


# ============================================== END USER FUNCTIONS DEFINED BELOW =============================================

def softmax(inp):
    bat, rows, cols = inp.shape

    for bat in range(bat):
        for row in range(rows):
            exp_arr = np.exp(inp[bat, row, :])
            inp[bat, row, :] = exp_arr/np.sum(exp_arr)



def cross_entropy_loss(logits, labels):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits/np.sum(exp_logits, axis=-1, keepdims=True)
    
    log_probs = -np.log(probs + 1e-9)
    loss = np.mean(np.sum(log_probs*labels, axis=-1))
    d_logits = (probs - labels)/logits.shape[0]
    
    return loss, d_logits

