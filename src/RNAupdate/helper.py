import scipy.io as sio
import numpy as np
import torch.nn as nn
import torch.utils.data as torchdata
import torch

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from sklearn.decomposition import PCA
from . import dataset_gen
from . import sudo_algo2
from . import encoder


batch_size = 32
device = 'cuda'

def plot_vec(l, xl = "", yl = ""):
    for cur in l:
        plt.plot([i for i in range(len(cur))], cur, '--') 
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.grid()
    
def dic_to_array(dic, lists_g):
    output = None
    for l in lists_g:
        cell_data = np.vstack([dic[g] for g in l])
        if output is None:
            output = cell_data
        else:
            output = np.vstack((output, cell_data))
    return output


def load_genes(f):
    res = []
    with open(f, 'r') as f:
        lines = f.readlines()
    res = [cur.strip('\n') for cur in lines]
    f.close()
    return res

def add_noise(X, scale):
    m, n = X.shape
    noise = np.random.normal(m, n)
    return X + noise * scale


def off_set(X):
    a, b = X.shape
    offset = (10**(-4)) * np.ones((a, b))
    X = X + offset
    return X
def log_output(X):
    return np.log2(X, out=np.zeros_like(X), where=(X!=0))

def get_index_ref_gene(g, gene_list):
    ref = []
    if g == 'z':
        ref = l_zonated 
    else:
        ref = l_circadian
    res = {}
    for cur in ref:
        for i in range(len(gene_list)):
            if gene_list[i] == cur:
                res[cur] = i
    return res 

# load data into a dic from gene to representation of corresponding key 
def load_data_key(key, files, path, total_gene):
#     path = '/n/home05/ysheng/Circadian-zonation/'
#     # map gene to cell
#     z_gene = load_genes(path + 'Z_gene.csv')
#     r_gene = load_genes(path + 'R_gene.csv')
#     z_r_gene = load_genes(path + 'Z+R_gene.csv')
#     zxr_gene = load_genes(path + 'ZxR_gene.csv')
    count = 0
    data = {}
#     for f in dic_struc:
    for f in files:
        load_path = path + 'Datasets/Profiles/ZT'+f+'.mat'
        cur_data = sio.loadmat(load_path)
#         print(cur_data.keys())
        cur_list = [temp[0].astype(str) for temp in cur_data['all_genes']]
        for i in range(len(cur_list)):
            g = cur_list[i][0].astype(str)
            if g in total_gene:
                if g in data:
    #                 print(type(data[g]))
    #                 print(type(cur_data['mat_norm'][i]))
                    data[g] = np.concatenate((data[g], cur_data[key][i]))
                else:
                    data[g] = cur_data[key][i]
            else: 
                if count <= 2000:
                    count += 1
                    if g in data:
                        data[g] = np.concatenate((data[g], cur_data[key][i]))
                    else:
                        data[g] = cur_data[key][i]
    #                 print(type(data[g])
                else:
                    if g in data:
                        data[g] = np.concatenate((data[g], cur_data[key][i]))
    return data

def remove_ref(l):
    res = []
    for g in l:
        if g not in l_zonated and g not in l_circadian:
            res.append(g)
    return res

def remove_0_rows(data):
    index = np.all(data == 0, axis=1)
    print(data[index[0]])
    return data[~index], index 

# output_mat_norm = dic_to_array(data_mat_norm, [r_gene, z_gene[0:600]])
def output_log_norm(output_mat_norm):
    a, b = output_mat_norm.shape
    offset = (10**(-4)) * np.ones((a, b))
    output_mat_norm_log = log_output(output_mat_norm + offset)
    output_mat_norm_log, index = remove_0_rows(output_mat_norm)
    output_mat_norm_log_norm = dataset_gen.normalized_row(output_mat_norm_log)
    return output_mat_norm_log_norm, index

def find_last_reconstruction_error(cur_ae, data, cur_weight, prev_weight_cell):
    temp = dataset_gen.cellDataset(data)
    dataset = torchdata.DataLoader(temp, batch_size, shuffle=False)
    diff_list = sudo_algo2.get_diff_matrix(cur_ae, dataset)
    diff = np.concatenate(tuple(diff_list), axis = 0)
    reweight_cell = prev_weight_cell[:, None] * diff
    exp_gene = np.sum(reweight_cell, axis = 0)
    return exp_gene

def get_encoder(cur_ae, data):
    result = []
    temp = dataset_gen.cellDataset(data)
    dataset = torchdata.DataLoader(temp, batch_size, shuffle=False)
    for x, index in dataset:
        x = x.to(device)
        # print(index)
        code, output = cur_ae(x.float())
#         print(code.shape)
        for c in code:
#             print(c)
            result.append(c.cpu().detach().numpy())
    return result

def get_output(cur_ae, data):
    result = []
    temp = dataset_gen.cellDataset(data)
    dataset = torchdata.DataLoader(temp, batch_size, shuffle=False)
    for x, index in dataset:
        x = x.to(device)
        code, output = cur_ae(x.float())
#         print(output.shape)
        for c in output:
#             print(c)
            result.append(c.cpu().detach().numpy())
    return result
def get_reconstruct(cur_ae, data):
    result = []
    temp = dataset_gen.cellDataset(data)
    dataset = torchdata.DataLoader(temp, batch_size, shuffle=False)
    for x, index in dataset:
        x = x.to(device)
        # print(index)
        code, output = cur_ae(x.float())
#         for c in output:
#             print(c)
        result.append((output - x.float()).cpu().detach().numpy())
    return result

def get_diff_matrix(cur_ae, data):
    temp = dataset_gen.cellDataset(data)
    dataset = torchdata.DataLoader(temp, batch_size, shuffle=False)
    diff_list = sudo_algo2.get_diff_matrix(cur_ae, dataset)
#     print(len(diff_list[0]))
    diff = np.concatenate(tuple(diff_list), axis = 0)
    return diff

def partial_average(g, codes):
    res = []
    for i in range(8):
        subset = [codes > i * 1/8][0] * [codes <= (i + 1) * 1/8][0]
#         print(subset)
        cur = np.sum(np.array(g)[subset])
        res.append(cur * 1/np.sum(subset))
    return res

def shift(g, i):
    ma=  np.max(g)
    mi = np.min(g)
    if (ma - mi) != 0:
        return [(cur - mi)/(ma - mi) for cur in g]
    else:
        return [0]*len(g)

def partial_average_shift_partition(codes):
    res = []
    for i in range(1, 9):
        low = np.max([0, i - 1])
        high = np.min([8, i + 2])
        subset = [codes >= low * 1/10][0] * [codes <= high * 1/10][0]
        res.append(subset)
    return res

def partial_average_shift_subset(g, subset, thre):
    res = []
    for i in subset:
        temp = np.array(g)[i]
        cur = temp[temp > thre]
        if len(cur > 0):
            res.append(np.sum(cur) * 1/len(cur))
        else:
#             print(len(temp))
            res.append(0)
    return res

def partial_average_shift(g, codes, thre):
    res = []
    for i in range(1, 9):
        low = np.max([0, i - 1])
        high = np.min([8, i + 2])
        subset = [codes >= low * 1/10][0] * [codes <= high * 1/10][0]
#         print(subset)
#         print(len(g))
#         print(len(subset))
        temp = np.array(g)[subset]
    
        cur = temp[temp > thre]
        if len(cur > 0):
            res.append(np.sum(cur) * 1/len(cur))
        else:
#             print(len(temp))
            res.append(0)
    return res

def time_base_avg(g,dic_time, m):
    res = []
    for i in range(4):
        cur_r = dic_time[i]
        temp = np.sum([np.sum(g[r[0]:np.min([r[1], m])]) for r in cur_r])
        l = np.sum([np.min([r[1], m]) - r[0] for r in cur_r])
        res.append(temp/l)
    return res

def histgram(gene_val, time_cell_num, t):
    ranges = time_cell_num[t]
    temp = np.concatenate([gene_val[r[0]:r[1]] for r in ranges])
    plt.hist(temp, bins = 50)
#     print(np.mean(transform_data(temp)))

import matplotlib.ticker as mtick

def plot_output_function(ae, index, low, high):
#     test_code = np.array([5 - i for i in range(50)]) * 3/50
    test_code = torch.Tensor([low + (i - 5) * (high - low)/ 50 for i in range(60)])
    print(test_code)
    function = []
#     ae = ae.to(device)
    for t in test_code:
        temp = ae.decoder(torch.tensor([t]).to(device)).to('cpu')
#         print(temp.size())
        function.append(temp.cpu().detach().numpy()) 
#         function.append(temp.detach().numpy()) 
#     plot_vec([np.array(function)[:, i] for i in index], "x", "f_g(x)")
    for i in index:
        plt.plot(test_code, np.array(function)[:, i], '--')
    plt.title("plot of decoder gene functions")
    plt.gca().xaxis.set_major_locator(plt.LinearLocator(5))
#     plt.xticks([i for i in range(len(test_code))], test_code)
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e')) 
    
    
    
def normalize_gene_ignore0(data, index):
#     record positive indexes
#     index = [data > 0][0]
    temp = data[index]
#     print(temp.shape)
#     if np.std(temp) == 0:
#         print(np.sum(index))
    if len(temp) > 0:
        s = np.std(temp)
#         for genes that have one non zero value
        if s == 0:
            data[index] = 1
        else:
            data[index] = (data[index] - np.mean(temp))/np.std(temp)
    
    return data
    
def coef_std(x):
#     temp = helper.shift(x, g)
    temp = x[x > 0]
    return np.std(temp)/np.mean(temp) if len(temp) > 0 else 0


# for computing ground truth mean gene expression
def compute_mean_gene_exp(data_mat, loc_dic):
    res = []
    for pos in loc_dic:
        mean = np.mean(data_mat[loc_dic[pos][0], :], axis = 0)
        res.append(mean)
    return np.array(res)
    