from . import helper
import numpy as np
from openpyxl import load_workbook
import anndata as ad
import scanpy as sc
import os
import pandas as pd
import matplotlib.pyplot as plt
from . import dataset_gen

import novosparc

# data: anndata (cell by gene)
def linear_novo(data, dim, n_neighbors_s, n_neighbors_t, eps):
    num_locations = dim
    locations_line = novosparc.gm.construct_line(num_locations=num_locations)
    # setup 
    tissue = novosparc.cm.Tissue(dataset=data, locations=locations_line)
    tissue.setup_smooth_costs(num_neighbors_s=n_neighbors_s, num_neighbors_t=n_neighbors_t)
#     alpha = 0 since no altas
    tissue.reconstruct(alpha_linear=0, epsilon=eps)
    return tissue

def circ_novo(data, dim, n_neighbors_s, n_neighbors_t, eps):
    num_locations = dim
    locations_circle = novosparc.gm.construct_circle(num_locations=num_locations)
    tissue = novosparc.cm.Tissue(dataset=data, locations=locations_circle)
    tissue.setup_smooth_costs(num_neighbors_s=n_neighbors_s, num_neighbors_t=n_neighbors_t)
    tissue.reconstruct(alpha_linear=0, epsilon=eps)
    return tissue

def loc_novo(data, dim, n_neighbors_s, n_neighbors_t, eps, location):
    num_locations = dim
#     locations_circle = novosparc.gm.construct_circle(num_locations=num_locations)
    
    tissue = novosparc.cm.Tissue(dataset=data, locations=location)
    tissue.setup_smooth_costs(num_neighbors_s=n_neighbors_s, num_neighbors_t=n_neighbors_t)
    tissue.reconstruct(alpha_linear=0, epsilon=eps)
    return tissue

def novo_ref(data, n_neighbors_s, n_neighbors_t, eps, location, atlas_matrix, markers_to_use, alpha_linear, dge_rep):
    tissue = novosparc.cm.Tissue(dataset=data, locations=location)
    tissue.setup_smooth_costs(dge_rep=dge_rep, num_neighbors_s=n_neighbors_s, num_neighbors_t=n_neighbors_t)
    tissue.setup_linear_cost(markers_to_use, atlas_matrix)
    tissue.reconstruct(alpha_linear=alpha_linear, epsilon=eps)
    return tissue

# for consistancy
def derivative(d, cell_dist):
    num_cell, dim = cell_dist.shape
    mean = d.dot(cell_dist * dim)
    
    deri = np.array([np.mean(mean[:, int(i * dim/8):int((i + 1)* dim/8)], axis = 1) for i in range(8)]).T
    return np.var(deri, axis = 1)


# find gene with smallest variance relative to its original variance w.r.t. uniform measure
def update_weight_quad(cur_w,d, cell_dist, lamb = 1):
    
    step_size = 0.2
    
    num_cell, dim = cell_dist.shape
    org_var = np.var(d, axis = 1)
#     compute current var
    mean = d.dot(cell_dist * dim)
    var = (d **2).dot(cell_dist * dim)- mean**2
    
#     pos_count = np.sum(d > 0, axis = 1)
#     pos_count = pos_count /np.mean(pos_count)
    
    var[org_var == 0] = 1
    org_var[org_var == 0] = 1
#     var[np.isnan(org_var)+ (org_var == 0)] = 1
#     org_var[np.isnan(org_var) + (org_var == 0)] = 1
#     print([org_var == 0])
    deri = derivative(d, cell_dist)
    
    exp_gene = lamb * (np.mean(var, axis = 1)/ org_var) + (1 - lamb) * (deri)
#     print(org_var)
#     plt.plot(1/org_var)
    
    res_gene = cur_w * np.exp((-1)*step_size * exp_gene) 
    tot_gene = np.sum(res_gene)
    
    return res_gene

# assume cur_w is over marker genes
def update_weight_lin(cur_w, d, cell_dist, atlas, lamb, len_marker):
    step_size = 0.1
    num_cell, dim = cell_dist.shape
    org_var = np.var(d, axis = 1)
    
    mean = d.dot(cell_dist * dim) 
    var = (d **2).dot(cell_dist * dim)- mean**2
    
#     var[org_var == 0] = 1
#     org_var[org_var == 0] = 1
    
    
    norm_atl = atlas/np.linalg.norm(atlas, axis = 1)[:, None]
#     print(norm_atl.shape)
    norm_mean = mean[0:len_marker]/np.linalg.norm(d, axis = 1)[:len_marker, None]
    
    temp2 = np.linalg.norm(norm_atl - norm_mean, 2, axis = 1)
#     pad = np.array([np.mean(temp2)] * (len(var) - len_marker))
#     pad = np.array([0] * (len(var) - len_marker))
#     temp = np.concatenate((temp2, pad)) * np.sum(cur_w[:len_marker])

    exp_gene = temp2
#     exp_gene = lamb * (np.mean(var, axis = 1)/ org_var) + (1 - lamb) * temp
    res_gene = cur_w * np.exp((-1)*step_size * exp_gene) 
#     tot_gene = np.sum(res_gene)
    return np.array(res_gene)


def round_zero(w):
    w[w < 1/len(w) * 0.9] = 0
    return w/np.sum(w)


def multi_weight_novo_param(init_w, dataset, num_itr, dim, n_nei_s, n_nei_t, eps, lamb):
    
    cur_w = init_w
    w_l = [cur_w]
    res_dic = {}
    raw_data = dataset.X.copy()
    for i in range(num_itr):
        print('cur itr is ' + str(i) + ' training with weight ')
#         plt.plot(cur_w)
#         plt.show()
#       input reweighted data
        dataset.X = raw_data * cur_w
        cur_tissue = linear_novo(dataset, dim, n_nei_s, n_nei_t, eps)
        cur_w = update_weight_quad(cur_w, dataset.X.T, cur_tissue.gw, lamb)
        cur_w = cur_w/np.sum(cur_w)
#         cur_w = round_zero(cur_w)
        res_dic[i] = cur_tissue
        w_l.append(cur_w)
    return res_dic, cur_w, w_l


def multi_weight_novo_param_circ(init_w, dataset, num_itr, dim, n_nei_s, n_nei_t, eps, lamb):
    
    cur_w = init_w
    res_dic = {}
    raw_data = dataset.X.copy()
    for i in range(num_itr):
        print('cur itr is ' + str(i) + ' training with weight ')
#         plt.plot(cur_w)
#         plt.show()
#       input reweighted data
        dataset.X = raw_data * cur_w
        cur_tissue = circ_novo(dataset, dim, n_nei_s, n_nei_t, eps)
        cur_w = update_weight_quad(cur_w, dataset.X.T, cur_tissue.gw, lamb)
#         cur_w = round_zero(cur_w)
        res_dic[i] = cur_tissue
    return res_dic, cur_w

def multi_weight_novo_param_loc(init_w, dataset, num_itr, dim, n_nei_s, n_nei_t, eps, lamb, location):
    
    cur_w = init_w
    res_dic = {}
    raw_data = dataset.X.copy()
    for i in range(num_itr):
        print('cur itr is ' + str(i) + ' training with weight ')
#         plt.plot(cur_w)
#         plt.show()
#       input reweighted data
        dataset.X = raw_data * cur_w
        cur_tissue = loc_novo(dataset, dim, n_nei_s, n_nei_t, eps, location)
        cur_w = update_weight_quad(cur_w, dataset.X.T, cur_tissue.gw, lamb)
        cur_w = cur_w/np.sum(cur_w)
#         cur_w = round_zero(cur_w)
        res_dic[i] = cur_tissue
    return res_dic, cur_w

def multi_weight_novo_ref(init_w, dataset, num_itr, n_nei_s, n_nei_t, eps, lamb, location, atlas_matrix, markers_to_use, alpha_linear, gene_list):
    
    cur_w = init_w
    
    res_dic = {}
    raw_data = dataset.to_df()[gene_list].copy()
    for i in range(num_itr):
        print('cur itr is ' + str(i) + ' training with weight ')
        plt.plot(cur_w)
#         plt.show()
#       input reweighted data
#         dataset.X = raw_data * cur_w
        dge_rep = raw_data * cur_w
        cur_tissue = novo_ref(dataset, n_nei_s, n_nei_t, eps, location, atlas_matrix, markers_to_use, alpha_linear, dge_rep)
        cur_w = update_weight_quad(cur_w, dge_rep.to_numpy().T, cur_tissue.gw, lamb)
        cur_w = cur_w/np.sum(cur_w)
        cur_w = round_zero(cur_w)
        res_dic[i] = cur_tissue
    return res_dic, cur_w


def multi_weight_novo_autotune(init_w, dataset, num_itr, n_nei_s, n_nei_t, eps, lamb, location, atlas_matrix, markers_to_use, alpha_linear, gene_list):
    
    cur_w = init_w
    res_dic = {}
    raw_data = dataset.to_df()[gene_list]
    org_atlas = atlas_matrix.copy()
    len_marker = len(markers_to_use)
    for i in range(num_itr):
        print('cur itr is ' + str(i) + ' training with weight ')
#         plt.plot(cur_w)
#         plt.show()
#       input reweighted data
#         dataset.X = raw_data * cur_w
        dge_rep = raw_data * cur_w
#         print(org_atlas.shape)
#         print(cur_w.shape)
        atlas = org_atlas
        alpha_linear = np.min((0.5 + np.sum(cur_w[:len_marker]), 1))
        print('cur alpha linear ' + str(alpha_linear))
        cur_tissue = novo_ref(dataset, n_nei_s, n_nei_t, eps, location, atlas, markers_to_use, alpha_linear, dge_rep)
#         print(len_marker)
        
        cur_w = update_weight_quad(cur_w, dge_rep.to_numpy().T, cur_tissue.gw, lamb)
        cur_w = cur_w/(np.sum(cur_w))
        cur_w[cur_w < np.mean(cur_w)] = 0
        cur_w = cur_w/(np.sum(cur_w))
#         cur_w2 = cur_w2/(np.sum(cur_w) + np.sum(cur_w2))
#         cur_w = round_zero(cur_w)
        res_dic[i] = cur_tissue
    return res_dic, cur_w


def multi_weight_novo_reweight_atl(init_w, dataset, num_itr, n_nei_s, n_nei_t, eps, lamb, location, atlas_matrix, markers_to_use, alpha_linear, gene_list, update_w2):
    w1 = []
    w2 = []
    cur_w = init_w
    cur_w2 = np.array([1/len(markers_to_use)] * len(markers_to_use))
    res_dic = {}
    raw_data = dataset.to_df()[gene_list]
    org_atlas = atlas_matrix.copy()
    len_marker = len(markers_to_use)
    for i in range(num_itr):
        print('cur itr is ' + str(i) + ' training with weight ')
#         plt.plot(cur_w)
#         plt.show()
#       input reweighted data
#         dataset.X = raw_data * cur_w
        dge_rep = raw_data * cur_w
#         print(dge_rep.to_numpy()[:, 0:60].T.shape)
#         print(dge_rep[gene_list[0:len_marker]].to_numpy().shape)
        atlas = org_atlas * cur_w2
        cur_tissue = novo_ref(dataset, n_nei_s, n_nei_t, eps, location, atlas, markers_to_use, alpha_linear, dge_rep)
        print(len_marker)
        
        cur_w = update_weight_quad(cur_w, dge_rep.to_numpy().T, cur_tissue.gw, lamb)
#         cur_w = update_weight7(cur_w, dge_rep.to_numpy().T, cur_tissue.gw, lamb)
        cur_w = cur_w/np.sum(cur_w)
        if update_w2:
            cur_w2 = update_weight_lin(cur_w2, dge_rep.to_numpy().T, cur_tissue.gw, atlas.T, lamb, len_marker)
    #         cur_w2 = update_weight6(cur_w2, dge_rep.to_numpy()[:, 0:60].T, cur_tissue.gw, atlas.T, lamb, len_marker)
            cur_w2 = cur_w2/np.sum(cur_w2)
#         cur_w = round_zero(cur_w)
        res_dic[i] = cur_tissue
        w1.append(cur_w)
        w2.append(cur_w2)
    return res_dic, w1, w2

