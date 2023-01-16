from . import encoder
from . import dataset_gen
from . import helper
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import pylab as pl
from IPython import display
import time 

# epochs = 10
batch_size = 64
# lr = 0.008
device= "cuda" if torch.cuda.is_available() else "cpu"

# multiplicative weight update
# def get_diff(autoencoder, dataset):
#     res = None
#     for x, index in dataset:
#         x = x.to(device)
#         code, output = autoencoder(x.float())
#         temp = output.cpu().data.detach().numpy() - x.numpy()
#         # temp_norm = np.linalg.norm(temp, 2, axis=0)
#         temp_norm = np.linalg.norm(temp, 2, axis=0)**2
# #         print(temp_norm.shape)
#         if res is None:
#             res = temp_norm
#         else:
#             res += temp_norm
#     # match weigthed MSE loss
#     # res = np.sqrt(res) 
# #     print(res)
#     return res


# # input: current autoencoder and the pytorch dataset to evaulate on
# # output new weight
# def update_weight(autoencoder, data, prev_weight, step_size):
#     dataset = torch.utils.data.DataLoader(data, batch_size, shuffle=True)
#     l = len(prev_weight)
# #     initalize result to be a vector of 1
# #     res = [1]*l
#     print('llendata is', len(data))
#     diff_vec = get_diff(autoencoder, dataset) * (1/len(data))
    
#     res = prev_weight*np.exp(-1  * step_size * diff_vec)
    
#     total = np.sum(res)

#     result = res/total
#     return result

def get_diff_matrix(autoencoder, dataset):
    res = []
    for x , index in dataset:
        x = x.to(device)
        code, output = autoencoder(x.float())
        temp = output.cpu().data.numpy() - x.cpu().data.numpy()
        res.append(np.array(temp)**2)
        
    return res

# def update_weight_gene_cell(autoencoder, data, prev_weight, prev_weight_cell, step_size):
#     dataset = torch.utils.data.DataLoader(data, batch_size, shuffle=False)

#     diff_list = get_diff_matrix(autoencoder, dataset)
#     diff = np.concatenate(tuple(diff_list), axis = 0)
#     # print(np.sum(diff))
#     reweight_cell = prev_weight_cell[:, None] * diff
#     exp_gene = np.sum(reweight_cell, axis = 0)
#     res_gene = prev_weight * np.exp((-1)*step_size * exp_gene)
#     # print(exp_gene)
#     tot_gene = np.sum(res_gene)
#     # print(tot_gene)

#     reweight_gene = prev_weight[None, :] * diff
#     exp_cell = np.sum(reweight_gene, axis = 1)
#     res_cell = prev_weight_cell * np.exp((-1)* step_size * exp_cell)
#     tot_cell = np.sum(res_cell)

#     return res_gene/tot_gene, res_cell/tot_cell

# matrix cell weight
def update_weight_gene_or_cell(autoencoder, data, prev_weight, prev_weight_cell, step_size, ifgene):
    dataset = torch.utils.data.DataLoader(data, batch_size, shuffle=False)
    diff_list = get_diff_matrix(autoencoder, dataset)
    diff = np.concatenate(tuple(diff_list), axis = 0)
    if ifgene:
#         reweight_cell = prev_weight_cell[:, None] * diff
        reweight_cell = prev_weight_cell * diff
        exp_gene = np.sum(reweight_cell, axis = 0)
        res_gene = prev_weight * np.exp((-1)*step_size * exp_gene)
        tot_gene = np.sum(res_gene)
        return res_gene/tot_gene
    else:
        reweight_gene = prev_weight[None, :] * diff
        exp_cell = np.sum(reweight_gene, axis = 1)
        res_cell = prev_weight_cell * np.exp((-1)*step_size * 10 * exp_cell)
        tot_cell = np.sum(res_cell)
        return res_cell/tot_cell
# =========================================================================================================================================

def update_weight_MSE(autoencoder, data, prev_weight, step_size):
    dataset = torch.utils.data.DataLoader(data, batch_size, shuffle=True)
    var_weight = torch.tensor(prev_weight, requires_grad = True)
    d = len(prev_weight)
    criterion = nn.MSELoss()
    

    result  = np.zeros(d)
    for x, index in dataset:
        m, d = x.shape
        weighted_x = var_weight * x
        code, output = autoencoder(weighted_x.float())
        cur_loss = criterion(output, weighted_x.float())
        grad = torch.autograd.grad(cur_loss, var_weight)
        # print(result)
        result += grad[0].detach().numpy() 

    exps =  result
    res = prev_weight * np.exp((-1) * step_size * exps)
    # print(res)
    total = np.sum(res)
    return res/total
             

# =========================================================================================================================================

def relative_entropy(p, q):
    return p * np.log(p/q)

# =========================================================================================================================================

def reweight_data(data, w):
    return data * w[:, None]

def reweight_data_cell(data, w):
    return data * w[None, :]

    
def multi_weight_weightedMSE(data, d, epoch, cod_dim, list_dims, step_size, num_round, init_weight, init_weight_cell, act_fn, early_stop = False):
#     inputs = ['data-dim', 'max epoch', 'code-dim', 'first-layer-dim: ', 'step-size']
    print('training with ' + 'data-dim: ' + str(d) + 'max epoch: ' + str(epoch) + 
         'code-dim: ' + str(cod_dim) + 'dimension of hidden layers are: ' + str(list_dims) + 'step-size: ' + str(step_size))

    #     add dynamic plot as data generates
#     plt.ion()
#     fig=plt.figure()
    # plt.axis([0,d,0,2/d])
    x = [i for i in range(d)]
    cur_weight = init_weight
    cur_weight_cell = init_weight_cell
    threshold = 1 / len(init_weight[init_weight > 0]) * (0.95)
    cur_ae = None
    loss = []
    loss_list = []
    test_error = []
    w_list = []
    encoding = {}
    itr = 0
#     total_round = 20
    total_round = num_round
    average_weight = np.copy(cur_weight)
    max_loss = np.inf
#     sample_block = 20 if early_stop else 10
    sample_block = 20
#     print('num sample is ' + str(sample_block))
#     store the autoencoder and weight that gives the lowest test error
#     test_ae = None
    test_w = [] 
    min_test_error = np.inf
    while (itr <= total_round):
        print('cur iteration is ' + str(itr))

        if len(data.shape) == 3:
            print("sythentic data")
            index = 0
            index2 = 0
            cur_data = encoder.get_dataset_from_list(data, index* 200 + 1, 200*index+ sample_block + 1)
            
            cur_valid = encoder.get_dataset_from_list(data, index2 * 200 + sample_block + 5, sample_block + 10 + index2 * 200)
        # cur_test = encoder.get_dataset_from_list(data, 450, 500)
        else:
#             real input
            s, t = data.shape
            valid_range = int(t * 0.9)
            print(valid_range)
            cur_data = dataset_gen.cellDataset(data[:, 0:valid_range])
            cur_valid = dataset_gen.cellDataset(data[:, valid_range:t])
#             print(cur_data.shape)
#             print(cur_valid.shape)
        
        cur_ae, cur_loss, cur_epoch_loss = encoder.training_weighted_MSE(cur_data, cur_valid, epoch, d, cod_dim, list_dims, cur_weight, cur_weight_cell, max_loss, None, act_fn, early_stop)
        # max_loss = cur_loss[-1]
        # print(max_loss)
        loss_list.append(cur_epoch_loss)
        cur_dataset = torch.utils.data.DataLoader(cur_data, batch_size, shuffle=True)
        # cur_loss = encoder.test_err_weighted(cur_ae, cur_dataset, cur_weight, cur_weight_cell)
        loss.append(cur_loss[-1])
        cur_codes = helper.get_encoder(cur_ae, data)
        encoding[itr] = cur_codes
        if np.mod(itr, 2) == 0 or np.mod(itr, 2) == 1:
            cur_weight = update_weight_gene_or_cell(cur_ae, cur_data, cur_weight, cur_weight_cell[:, :valid_range].T, step_size, True)
#             cur_weight = update_weight_gene_or_cell(cur_ae, cur_data, cur_weight, cur_weight_cell, step_size, True)
        else:
            cur_weight_cell = update_weight_gene_or_cell(cur_ae, cur_data, cur_weight, cur_weight_cell, step_size, False)
            # print(np.sum(cur_weight_cell[0:400000]))
        print(np.sum(cur_weight))
        cur_loss = encoder.test_err_weighted(cur_ae, cur_dataset, cur_weight, cur_weight_cell)
        

        max_loss = cur_loss
        w_list.append(np.copy(cur_weight))
        
        loss.append(cur_loss)
#         pl.plot(x, cur_weight, 'o--') 
#         display.clear_output(wait=True)
#         display.display(pl.gcf())
#         time.sleep(1.0)
        
        
        itr += 1
    return cur_ae, cur_weight, w_list, loss, test_error, test_w, cur_weight_cell, loss_list, encoding
    
    
# def multi_weight(data, d, epoch, cod_dim, first_layer_dim, step_size, num_round):
#     # weight = np.ones(d) * (1/d)
#     weight = np.array([0.1] + [0.9/199]*199)
#     return multi_weight_init(data, d, epoch, cod_dim, first_layer_dim, step_size, num_round, weight)


def run_exp(output_mat, epoch, code_dim, list_dims, step_size, num_itr, act_fn, init_subset):
    data_dim, cell_dim = output_mat.shape
    pos_count = np.sum(output_mat > 0, axis = 1)
    l_v = data_dim
    l_z = init_subset
    init_weight =  np.array([1/l_z] * l_z + [0] * (l_v - l_z))
    inv_pos_count = np.array([1/x if (x > 0 * cell_dim) else 1 for x in pos_count])
    init_cell_weight = np.ones((data_dim, cell_dim)) * inv_pos_count[:, None]
    res = multi_weight_weightedMSE(output_mat, data_dim,epoch, code_dim, list_dims, step_size, num_itr, init_weight,init_cell_weight, act_fn)
    return res
    
# =========================================================================================================================================

def round(w, threshold):
   
    w[w < threshold] = 1/len(w) * 0.01
    return w/np.sum(w)


def find_subset(w):
    threshold = np.mean(w[w > 1/len(w) * 0.01]) * 0.95
    return np.array([w > threshold], dtype = int)

def True_positive(w, true_w):
    real = np.sum(true_w)
    computed = np.sum(w*true_w)
#     print(w*true_w)
    return computed/real

def False_positive(w, true_w):
    n = len(w)
    complement = np.ones(n) - true_w
    return True_positive(w, complement)


