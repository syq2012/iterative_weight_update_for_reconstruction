import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from . import dataset_gen

import random

seed = 121
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# 
# Settings
epochs = 30
batch_size = 64
# batch_size = 10

# lr = 0.008
device = "cuda" if torch.cuda.is_available() else "cpu"
# DataLoader
# read the first l set of samples from file, 
# each set of sample is of size d by n
def get_dataset(file, d, n, l, h):
    raw_data = dataset_gen.read(file, d, n)
    # print(raw_data[0].shape)
    raw_inputs = np.concatenate(tuple([raw_data[i] for i in range(l, h)]), axis = 1)
    dataset = dataset_gen.cellDataset(raw_inputs)
    return dataset
# input is a list of numpy matrices and the range of data to read
def get_dataset_from_list(raw_data, l, h):
#     raw_data = dataset_gen.read(file, d, n)
    # print(raw_data[0].shape)
    raw_inputs = np.concatenate(tuple([raw_data[i] for i in range(l, h)]), axis = 1)
    dataset = dataset_gen.cellDataset(raw_inputs)
    return dataset  
# load datas as pytorch dataset
def get_torchdataset(datas):
    return torch.utils.data.DataLoader(datas, batch_size,
            shuffle=True)
# =========================================================================================================================================


class custom_discrete(nn.Module):
    #the init method takes the parameter:
    def __init__(self, dim, output_dim):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim

    #the forward calls it:
    def forward(self, x):
        multiplier = torch.tensor([2 ** i for i in range(self.dim)]).to(device)
        temp = torch.sum(x * multiplier, dim = 1)
        return torch.stack([temp**i for i in range(self.output_dim)], dim = 1).to(device)
    
# =========================================================================================================================================

# Model structure
class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        print(kwargs)
        data_dim = kwargs["input_shape"]
        code_dim = kwargs["code_dim"]
        list_dim = kwargs["list_dim"]
        activate = kwargs["act_fn"]
        complete_layer_dim = [data_dim]+list_dim
        layer_list = []
        for i in range(len(complete_layer_dim) - 1):
          layer_list.append(nn.Linear(complete_layer_dim[i], complete_layer_dim[i + 1]))
          if activate == 'tanh' or activate == 'tanh_enc':
            layer_list.append(nn.Tanh())
          elif activate == 'relu' or activate == 'relu_enc':
            layer_list.append(nn.ReLU6())
        layer_list.append(nn.Linear(complete_layer_dim[-1], code_dim))

        decoder_layers = [nn.Linear(code_dim, complete_layer_dim[-1])]
        for i in range(1, len(complete_layer_dim)):
          if activate == 'tanh':
            layer_list.append(nn.Tanh())
          elif activate == 'relu':
            layer_list.append(nn.ReLU6())
          decoder_layers.append(nn.Linear(complete_layer_dim[-i], complete_layer_dim[-(i + 1)]))
        self.encoder = nn.Sequential(*layer_list)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, inputs):
        codes = self.encoder(inputs)
        decoded = self.decoder(codes)

        return codes, decoded

# =========================================================================================================================================

def MSE_transpose(output, target):
    return (1/batch_size) * torch.sum((output - target)**2)

def training(train_dataset, valid_dataset, epochs, input_length, code_dim, first_layer_dim, act_fn):
    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #  get dataloader 
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    valid_data = torch.utils.data.DataLoader(valid_dataset, batch_size, shuffle=True)
    
    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    list_dimension = [first_layer_dim, int(first_layer_dim / 2)]
    model = AE(input_shape=input_length, code_dim = code_dim, first_layer_dim = first_layer_dim, list_dim = list_dimension, act_fn = act_fn).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    learning_rate= 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_list = []
    # mean-squared error loss
    criterion = nn.MSELoss()
    
#     for early stopping
    valid_error = np.inf
    res_ae = None
    for epoch in range(epochs):
        loss = 0
        for x, index in train_data:
            x = x.to(device)
            
            code, outputs = model(x.float())
            train_loss = criterion(outputs, x.float())
#             train_loss = weighted_MSELoss(outputs, x.float(), weight)
            
            train_loss.backward()
            optimizer.step()
        
            loss += train_loss.item()
        print('[{}/{}] Loss:'.format(epoch+1, epochs), loss)
        loss_list.append(loss)
#         loss_list.append(train_loss.item())
        cur_valid_err = test_err(model, valid_data)
        if cur_valid_err >= valid_error:
            break
        else:
            valid_error = cur_valid_err
            res_ae = model
            loss_list.append(train_loss.item())
    return res_ae, loss_list


def weighted_MSELoss(output, target, weight, weight_cell):
    weight_tensor = torch.tensor(weight).to(device)
    
    (m, n) = output.size()
    return torch.sum(torch.tensor(weight_cell).to(device)[:, None] * (weight_tensor *  (output - target) ** 2))

# matrix cell weight    
def weighted_MSELoss_ignore0(output, target, weight, weight_cell):
    weight_tensor = torch.tensor(weight).to(device)
    temp = (output - target)
    temp[target == 0] = 0
    return torch.sum(torch.tensor(weight_cell.T).to(device) * (weight_tensor *  temp ** 2))
    

import torch.nn.functional as F

def sparse_loss(model, data):
    model_children = list(model.children())
    loss = 0
    values = data
    for i in range(len(model_children)):
        values = (model_children[i](values))
        loss += torch.mean(torch.abs(values))
    return loss * (1/len(model_children))

def training_weighted_MSE(train_dataset, valid_dataset, epochs, input_length, code_dim, list_dims, weight, weight_cell, max_loss, init_ae, act_fn, early_stop = False):
    
    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #  get dataloader 
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    # valid_data = torch.utils.data.DataLoader(valid_dataset, batch_size, shuffle=True)
    
    valid_data = torch.utils.data.DataLoader(valid_dataset, batch_size, shuffle=True)
    
    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    if init_ae == None:
        model = AE(input_shape=input_length, code_dim = code_dim, list_dim = list_dims, act_fn = act_fn).to(device)
    else:
        model = init_ae

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    loss_list = []
    epoch_loss = []
    # mean-squared error loss
#     criterion = nn.MSELoss()
#     criterion = weighted_MSELoss()
    
#     for early stopping
    valid_error = np.inf
    res_ae = None
    min_train_loss = np.inf
    patience = 0
    
#     print('max_loss is' + str(max_loss))
#     res_loss = None
    for epoch in range(epochs):
        loss = 0
        for x, index in train_data:
            x = x.to(device)
            # print(index)
            code, outputs = model(x.float())
            
            train_loss = weighted_MSELoss_ignore0(outputs.to(device), x.float(), weight, weight_cell[:, index]) 
            
            
            train_loss.backward()
            optimizer.step()
        
            loss += train_loss.item()
 
#         print('[{}/{}] Loss:'.format(epoch+1, epochs), loss)
        epoch_loss.append(loss)
        
        
#     matrix value cell weigth
        valid_range = len(valid_dataset)
        weight_cell_valid = weight_cell[:, (-1) * valid_range:]
        cur_valid_err = test_err_weighted(model, valid_data, weight, weight_cell_valid)
            
        if early_stop:
          if max_loss < np.inf:
              # record the best model so far. min_train_loss > max_loss
              if loss <= min_train_loss:
                  print(min_train_loss)
                  min_train_loss = loss
                  res_ae = model
                  loss_list.append(loss)
              if loss < max_loss and patience > 20:
                  # print(res_ae == None)
                  return res_ae, loss_list, epoch_loss
              else:
                patience += 1
          else:
            
            if cur_valid_err >= valid_error:
              if patience > 20:
                return res_ae, loss_list, epoch_loss
              else:
                patience += 1 
            else:
                valid_error = cur_valid_err
                res_ae = model
                loss_list.append(loss)
        else:
          res_ae = model
          loss_list.append(loss)
    return res_ae, loss_list, epoch_loss

# =========================================================================================================================================


# compute test error of a given autoencoder 
def test_err(autoencoder, data_test):
    criterion = nn.MSELoss()
    res = []
#     counter = 0
    for x, index in data_test:
#         if counter <= r:
        x = x.to(device)
        code, outputs = autoencoder(x.float())
        test_loss = criterion(outputs, x.float())
#         print(type(test_loss))
        res.append(test_loss.item())
#         counter += batch_size
    return np.mean(res)

def test_err_weighted(autoencoder, data_test, weight, weight_cell):
    
    res = []
    for x, index in data_test:
#         if counter <= r:
        x = x.to(device)
        code, outputs = autoencoder(x.float())
        test_loss = weighted_MSELoss_ignore0(outputs, x.float(), weight, weight_cell[:, index])
    

        res.append(test_loss.item())
        
    return np.sum(res)

# Save
def save_autoencoder(model):
    torch.save(model, 'autoencoder.pth')
