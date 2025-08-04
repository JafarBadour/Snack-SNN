import cupy as cp
from cupyx.scipy.sparse import coo_matrix



def transform_coo2scr(indices_a, indices_b, values, matrix_shape):
    """
    Transform a COO matrix to a CSR matrix.
    """
    # Create COO matrix using cupy
    coo = coo_matrix((values, (indices_a, indices_b)), shape=matrix_shape)
    
    # Convert to CSR format
    csr = coo.tocsr()
    
    # Extract CSR components
    indptr = csr.indptr
    indices = csr.indices  
    data = csr.data
    
    return indptr, indices, data

