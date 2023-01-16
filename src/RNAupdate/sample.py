import numpy as np
import itertools

# import sudo_algo

# functions for sample random points as data set

# input: n number of samples, range: the range of random sample 
def sample_x(n, range):
	return np.random.uniform(0, range, n);

# input: npoints: number of samples
def sample_spherical(npoints, ndim=2):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec
def sample_swiss(num_cell):
	temp = np.random.uniform(low = 0, high = 1, size = n)
	x = temp * np.sin(temp)
	y = temp * np.cos(temp)
	return np.row_stack((x, y))

def sample_circle(num_cell):
	temp = np.random.uniform(low = 0, high = 2 * np.pi, size = num_cell)
	x = np.sin(temp)
	y = np.cos(temp)
	return np.row_stack((x, y))

# input: n number of samples
def sample_disk(n):
	r = np.random.uniform(low=0, high=1, size=n)  # radius
	theta = np.random.uniform(low=0, high=2*np.pi, size=n)  # angle
	x = np.sqrt(r) * np.cos(theta)
	y = np.sqrt(r) * np.sin(theta)
	return np.row_stack((x, y))

def sample_torus(n, R, r):
	size = (2, n)
	angles = np.pi * 2 * np.random.uniform(0, 1, size)
	res = np.zeros((3, n))
	for i in range(n):
		theta = angles[0][i]
		phi = angles[1][i]
		temp = R + r * np.cos(theta)
		A = np.array([temp * np.cos(phi), temp * np.sin(phi), r * np.sin(theta)])
	return res

# sample n samples from gaussian with mean \sim unif([0, 1]^d) and covariance = identity  
def sample_gaussian_mean(n, d, mean):
	covar = np.identity(d)
	# print(mean)
	res = np.random.multivariate_normal(mean, covar, n)
	return res.T

def sample_gaussian(n, d):
	mean = np.random.rand(d)
	return sample_gaussian_mean(n, d, mean)

def sample_gaussian_0mean(n, d):
	mean = np.zeros(d)
	return sample_gaussian_mean(n,d,mean)

# return matrix where each term in the matrix is sampled uniformly from [0, 1]
def sample_noise(m, n):
	return np.random.rand(m, n)


def sample_gaussian_noise(d, n):
	return sample_gaussian_0mean(n, d)
# =========================================================================================================================================

def get_subset(d, k):
	# add empty set
	res = [()]
	# find base
	base = [i for i in range(d)]
	for i in range(k):		
		# res += list(itertools.combinations(base, i + 1))
		res += list(itertools.combinations_with_replacement(base, i + 1))
	return res


# each row is a sample
def lift(x, k):
	(m, n) = x.shape
	# add x^0 as the first row
	# pad_x = np.vstack([np.ones(n), x])
	monomials = get_subset(m, k)
	res = np.array([np.prod(x[s, :], axis = 0) for s in monomials])
	# print(res.shape)
	return res

# generate eval of x on d random polynomials
def random_polymonial(data, k, d):
	lift_x = [lift(np.array(x), k) for x in data]
	(m, n) = lift_x[0].shape
	coeff = np.random.rand(d, m)
	res = [coeff @ x for x in lift_x]
	# eval_polynomial(data, k, d, coeff)
	return lift_x, res, coeff

def eval_polynomial(data, k, d, coeff):
	lift_x = [lift(np.array(x), k) for x in data]
	(m, n) = lift_x[0].shape
	res = [coeff @ x for x in lift_x]
	return lift_x, res
	# lift_x = lift(x, k)
	# (m, n) = lift_x.shape
	# # coeff = np.random.rand(d, m)
	# res = coeff @ lift_x
	# print(res.shape == (d, n))
	# return res, coeff, lift_x


# =========================================================================================================================================

def combine_samples(T):
	return np.concatenate(T, axis = 0)

# take a normal gen_function, noisy matrix's dimension and intensity
# and noise_location \in {0, 1} where 0 means before, 1 means after
def pad_noise(gen_function, noise_dim, noise_intensity, noise_location):
	X = gen_function()
	(d, n) = X.shape
	N = sample_noise(noise_dim, n) * noise_intensity
	temp_N = sample_noise(d, n) * noise_intensity
	if noise_location == 0:
		return np.concatenate((N, X + temp_N), axis = 0)
	elif noise_location ==1:
		res = np.concatenate((X+ temp_N, N), axis = 0)
		# print(res.shape)
		return res
	else:
		print("unknown location")
		return np.zeros(1)
	 

# randomly permute rows of given data
# returns new order of rows and the permuted data
def shuffle_genes(X, m):
	# get new order of rows
	# new_order[i] = a means ath column in X is moved to ith column
	new_order = [i for i in range(m)]
	np.random.shuffle(new_order)

	# change order of rows
	result = X[new_order, :]
	return new_order, result

# randomly permute the columns of given data
def shuffle_cells(X, n):
	new_order, result = shuffle_genes(X.T, n)
	return new_order, result.T

def naive_encoder(k, m, noise):
	temp = np.block([np.identity(k), np.zeros((k, m - k))])
	if not noise == 0:
		(a, b) = temp.shape
		temp += np.random.rand(a, b) * noise
	return temp 

def naive_decoder(k, m, noise):
	result = naive_encoder(k, m, noise)
	return result.T

