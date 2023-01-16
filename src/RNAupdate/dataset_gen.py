import torch
import numpy as np
from numpy import savetxt
from numpy import loadtxt
from io import StringIO
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os.path

from torch.utils.data import TensorDataset, DataLoader

# get data generating function
from . import sample
# this is to generate dataset for future testing

# e.g. write(lambda: fun(arg), n)
def write(gen_fun, num_sample, name):
	result = [gen_fun().flatten() for i in range(num_sample)]
	f_name = name + '.csv'
	path = ''
	savetxt(path + f_name, result, delimiter= ',')

def read(file_name, d, n):
# 	path = '/content/drive/MyDrive/Colab Notebooks/'
	path = ''    
	data = loadtxt(path + file_name, delimiter = ',')
	result = [data[i].reshape((d, n)) for i in range(len(data))]
	return result

# generate num_sample set of samples, each set contains num_cell number of data point of 
# dimension k
def gen_sphereical(num_sample, num_cell,  k):
	name = 'sphere' + '_' + str(num_sample) + '_'  + str(num_cell) + '_'  +str(k)
	write(lambda: sample.sample_spherical(num_cell, k), num_sample, name)

# generating cell from a 1d circle
def gen_circle(num_sample, num_cell):
	name = 'circle' + '_' + str(num_sample) + '_'  + str(num_cell) 
	write(lambda: sample.sample_circle(num_cell), num_sample, name)

# generating data from a 1d swiss roll
def gen_swiss(num_sample, num_cell):
	name = 'swissroll' + '_' + str(num_sample) + '_'  + str(num_cell)
	write(lambda: sample.sample_swiss(num_cell), num_sample, name)

# generate torus of outer radius R and inner radius r
def gen_torus(num_sample, num_cell_toru, R, r):
	name = 'torus' + '_'  + str(num_sample)  + '_'  + str(num_cell_toru) + '_'  +str(R) + '_'  +str(r)
	write(lambda: sample.sample_torus(num_cell_toru, R, r), num_sample, name)

# generate disk of random radius
def gen_disk( num_sample, num_cell):
	name = 'disk' + '_'  + str(num_sample) + '_'  + str(num_cell)
	write(lambda: sample.sample_disk(num_cell), num_sample, name)

def gen_noise(num_sample, m, n):
	name = 'uniformnoise' + '_' + str(num_sample) + '_' + str(m) + '_' + str(n)
	write(lambda: sample.sample_noise(m, n), num_sample, name)

def gen_gaussain(num_sample, num_cell, dimension):
	name = 'gaussian' + '_' + str(num_sample) + '_' + str(num_cell) + '_' + str(dimension)
	write(lambda: sample.sample_gaussian(num_cell, dimension), num_sample, name)

def gen_gaussian_noise(num_sample, num_cell, dimension):
	name = 'gaussiannoise' + '_' + str(num_sample) + '_' + str(num_cell) + '_' + str(dimension)
	write(lambda: sample.sample_gaussian_noise(dimension, num_cell), num_sample, name)

def gen_gaussian_0mean(num_sample, num_cell, dimension):
	name = 'gaussian_0mean' + '_' + str(num_sample) + '_' + str(num_cell) + '_' + str(dimension)
	write(lambda: sample.sample_gaussian_0mean(num_cell, dimension), num_sample, name)



def gen_sphere_with_noise(num_sample, num_cell, k, noise_dimension, noise_intensity):
	lst = ['spherewithnoise', 'numsample', str(num_sample), 'num_cell', str(num_cell), 'noise_dim', str(noise_dimension), 'noise_intensity', str(noise_intensity)]
	name = '_'.join(lst)
	write(lambda: sample.pad_noise(lambda: sample.sample_spherical(num_cell, k), noise_dimension, noise_intensity, 1), num_sample, name)

def gen_torus_with_noise(num_sample, num_cell, R, r, noise_dimension, noise_intensity):
	lst = ['toruswithnoise', 'numsample', str(num_sample), 'num_cell', str(num_cell), 'noise_dim', str(noise_dimension), 'noise_intensity', str(noise_intensity)]
	name = '_'.join(lst)
	write(lambda: sample.pad_noise(lambda: sample.sample_torus(num_cell, R, r), noise_dimension, noise_intensity, 1), num_sample, name)

def gen_disk_with_noise(num_sample, num_cell, noise_dimension, noise_intensity):
	lst = ['diskwithnoise', 'numsample', str(num_sample), 'num_cell', str(num_cell), 'noise_dim', str(noise_dimension), 'noise_intensity', str(noise_intensity)]
	name = '_'.join(lst)
	write(lambda: sample.pad_noise(lambda:sample.sample_disk(num_cell), noise_dimension, noise_intensity, 1), num_sample, name)



# generate data valuated on d2 random polynomial of degree k
# original data of dimension d 
# returns coefficients of the polynomials (each row represent the correspoinding poly)
def gen_polynomial_data(file_name, d, n, k, d2):
	name = file_name + '.csv'
	cur_data = read(name, d, n)
	lifted, values, coeff = sample.random_polymonial(cur_data, k, d2)
	x_file = file_name + '_' + 'degree' + '_' + str(k) + '.csv'
	savetxt(x_file , [cur.flatten() for cur in lifted], delimiter= ',')
	new_name = file_name + '_num_poly_' + str(d2) + '_degree_' + str(k) + '.csv'
	savetxt(new_name, [cur.flatten() for cur in values], delimiter = ',')
	return coeff


# =========================================================================================================================================

# functions for combine data from two files, i.e. stacking or padding 
def get_dim(l):
    res = 3
    if l[0] == 'sphere':
        res = int(l[3])
    return res
# assume input datasets have the same number of samples 
# concat and add noise to two datasets

# change name
def combine_dataset(f1, f2, noise):
    l1 = f1.split('_')
    l2 = f2.split('_')
    l3 = noise.split('_')
    
    dim1 = get_dim(l1)
    dim2 = get_dim(l2)
#     print(dim1)
#     print(dim2)
#     print(l3[2] != l1[1])
    if (l1[1] != l2[1]) or (l1[2] != l2[2]) or (int(l3[2]) != (dim1 + dim2)) or l3[1] != l1[1]:
        print("data number not matched")
        return
    else:        
        raw1 = dataset_gen.read(f1 + '.csv', dim1, int(l1[2]))  
        raw2 = dataset_gen.read(f2+ '.csv', dim2, int(l2[2]))
        rawn = dataset_gen.read(noise+ '.csv', int(l3[2]), int(l3[3]))
        
        temp1 = [np.pad(x, ((0, dim2), (0, 0)), 'constant') for x in raw1]
        temp2 = [np.pad(x, ((dim1, 0), (0, 0)), 'constant') for x in raw2]
        res = [np.concatenate((t1, t1), axis = 1) for t1, t2 in zip(temp1, temp2)]
        return res, rawn

def stack_dataset(f1, f2, noise):
    l1 = f1.split('_')
    l2 = f2.split('_')
    l3 = noise.split('_')
    
    dim1 = get_dim(l1)
    dim2 = get_dim(l2)
    
    if (l1[1] != l2[1]) or (l1[2] != l2[2]) or (int(l3[2]) != dim1 + dim2) or l3[1] != l1[1]:
        print("data number not matched")
        return
    else:        
        raw1 = read(f1 + '.csv', dim1, int(l1[2]))  
        raw2 = read(f2+ '.csv', dim2, int(l2[2]))
        rawn = read(noise+ '.csv', int(l3[2]), int(l3[3]))
        
#         temp1 = [np.pad(x, ((0, dim2), (0, 0)), 'constant') for x in raw1]
#         temp2 = [np.pad(x, ((dim1, 0), (0, 0)), 'constant') for x in raw2]
        res = [np.concatenate((t1, t2), axis = 0) for t1, t2 in zip(raw1, raw2)]
        return res, rawn

def stack_dataset_dim(f1, f2, noise, d1, d2):
    l1 = f1.split('_')
    l2 = f2.split('_')
    l3 = noise.split('_')
    
#     dim1 = get_dim(l1)
#     dim2 = get_dim(l2)
    
    if (l1[1] != l2[1]) or (l1[2] != l2[2]) or (int(l3[2]) != d1 + d2) or l3[1] != l1[1]:
        print("data number not matched")
        return
    else:        
        raw1 = dataset_gen.read(f1 + '.csv', d1, int(l1[2]))  
        raw2 = dataset_gen.read(f2+ '.csv', d2, int(l2[2]))
        rawn = dataset_gen.read(noise+ '.csv', int(l3[2]), int(l3[3]))
        
#         temp1 = [np.pad(x, ((0, dim2), (0, 0)), 'constant') for x in raw1]
#         temp2 = [np.pad(x, ((dim1, 0), (0, 0)), 'constant') for x in raw2]
        res = [np.concatenate((t1, t2), axis = 0) for t1, t2 in zip(raw1, raw2)]
        return res, rawn

def get_dim(f):
	l = f.split('_')
	if l[0] == 'torus':
		return 3
	elif l[0] == 'disk' or l[0] == 'circle':
		return 2 
	elif l[0] == 'uniformnoise':
		return l[-2]
	elif l[-1] == 'padweights':
		return int(l[-2]) + 1
	else:
		return l[-1]

def get_num_sample(f):
	l = f.split('_')
	return l[1]

def stack_list(list_file, list_data_dim, num_cell):
	
	temp = [read(list_file[i] + '.csv', list_data_dim[i], num_cell) for i in range(len(list_file))]
	# for each matrix, normalized each row
	raw = [[normalized_row(np.array(r)) for r in cur] for cur in temp]
	
	# stack those matrices ontop of each other 
	r = np.min([len(r) for r in raw])
	res = [np.concatenate([r[i] for r in raw], axis = 0) for i in range(r)]

	return res


def stack_polynoimal(list_file, list_degree, list_num, num_sample, num_cell):
	l = len(list_file)

	if len(list_degree) != l or len(list_num) != l:
		print('input length not matched')

# 	noise_dim = np.sum(list_num)
# 	noise_name = 'gaussiannoise_' + str(num_sample) + '_' + str(num_cell) + '_'+str(noise_dim)
# 	if not os.path.isfile(noise_name + '.csv'):
# #         print('here')
# 		# gen_noise(num_sample, noise_dim, num_cell)
# 		gen_gaussian_noise(num_sample, num_cell, noise_dim)

	list_name = [list_file[i] + '_num_poly_' + str(list_num[i]) + '_degree_' + str(list_degree[i]) for i in range(l)]

	# if file does not exist, generate one
	for i in range(l):
# 		if not os.path.isfile('/content/drive/MyDrive/Colab Notebooks/' + list_name[i] + '.csv'):
		if not os.path.isfile(list_name[i] + '.csv'):
			print('missing' + list_name[i])
			cur_file = list_file[i]
			gen_polynomial_data(cur_file, int(get_dim(cur_file)), num_cell, list_degree[i], list_num[i])

	data = stack_list(list_name, list_num, num_cell)
	# noise = read(noise_name + '.csv', noise_dim, num_cell)
	noise = []
	return data, noise


# each column of input matrix is a cell/data
# normalized each row (gene) so that it has mean 0 and var 1
def normalized_row(X):
	mean = np.mean(X, axis = 1)
	var = np.sqrt(np.var(X, axis = 1))

	(m, n) = X.shape
	if (len(mean) != m):
		print('wrong length')

	temp = X - mean[:, None]
	return temp / (var[:, None])

def normalized_row_nan(X):
	X[X == 0] = 'nan'
	mean = np.nanmean(X, axis = 1)
	var = np.sqrt(np.nanvar(X, axis = 1))

	(m, n) = X.shape
	if (len(mean) != m):
		print('wrong length')

	temp = X - mean[:, None]
	temp[np.isnan(temp)] = 0
# 	print(temp)
# 	print(var)
	return temp / (var[:, None])

def standard_gaussian_pdf(X):
	norm = np.linalg.norm(X, 2, axis = 0)**2
	temp = np.exp(- (1/2) * norm)
	return 1 / (np.sqrt(2 * np.pi)) * temp

def pad_guassian_weight(file, dim, num_cell):
	samples = read(file + '.csv', dim, num_cell)
	padded_samples = [np.concatenate((cur, [standard_gaussian_pdf(cur)]), axis = 0).flatten() for cur in samples]

	new_name = file + '_padweights.csv'
	savetxt(new_name, padded_samples, delimiter= ',')

# =========================================================================================================================================
def pad_cell(list_data1, list_data2):
	return [np.concatenate((list_data1[i], list_data2[i]), axis = 1) for i in range(len(list_data1))]

def combine_list(l1, l2, range1, range2):
    res = [np.concatenate((l1[i][range1[0]:range1[1], :], l2[i][range2[0]:range2[1], :]), axis = 0) for i in range(len(l1))]
    return res
# =========================================================================================================================================
# construct dataset for pytorch 
# input: matrix with column being a cell 
class cellDataset(torch.utils.data.Dataset):
	# data is a array of matrices 
	def __init__(self, data):
		# parse data into tensor 
		d, n = data.shape
		self.data = [torch.from_numpy(data[:,i]) for i in range(n)]
#         get each row as a data point for reconstruction 
# 		self.data = [torch.from_numpy(data[i,:]) for i in range(d)]

	def __len__(self):
		return len(self.data)
	def __getitem__(self, index):
		return self.data[index], index


def construct_cell_dataset(data):
	return cellDataset(data)


	
	
