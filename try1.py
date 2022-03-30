import numpy as np
import scipy.sparse as sp


def normalize_adj(adj):
	"""Symmetrically normalize adjacency matrix."""
	adj = sp.coo_matrix(adj)
	rowsum = np.array(adj.sum(1))
	d_inv_sqrt = np.power(rowsum, -0.5).flatten()
	d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
	d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
	return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo().toarray()




coo0 = np.array([[1, 2, 3],
                [1,5,6],
                [7,2,4]])
print(coo0)

print('-----------------------------------')
coo1 = normalize_adj(coo0)
print(coo1)

print('-----------------------------------')
coo2 = coo1.toarray()
print(coo2)












