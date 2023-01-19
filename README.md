
To run the algorithm with AE:
  1. load scRNA data into a matrix of genes x cells
  2. Specify AE architecture by chosing
      a. max_epoch := number of epochs to run
      b. code dimension of AE
      c. hidden layers width. e.g [64, 32] would tell the algorithm to use AE with hidden layers 64 -> 32 -> code_dim 
      d. act_fun: could be 'tanh', 'relu', 'tanh_enc', 'relu_enc' or none. No act_fn means linear activation function is used
  3. choose the step size for weight update, typically 0.1
  4. run sudo_algo2.run_exp(data_mat, max_epoch, code_dim, hidden_layers, step_size, num_iter, act_fn, data_dim)
  
To run the algorithm with novoSpaRc:
  1. choose initial configuration for novoSpaRc (n_s := num_nei_s, n_t := num_nei_t, eps := epsilon (for sinkhorn))
  2. choose an initial weight over genes. Typically uniform  
  4. run novo_weight.multi_weight_novo_param(init_weight, filtered_data, num_itr, dim, n_s, n_t, eps, 0.5)
 
[tutorial.ipynb](https://github.com/syq2012/cleaned_up_rna/blob/main/tutorial.ipynb) is an jupyter notebook containing example code for both AE and novoSpaRc. 

Codes for generating Synthetic dataset is at [python script](https://github.com/syq2012/cleaned_up_rna/blob/main/src/RNAupdate/dataset_gen.py)
