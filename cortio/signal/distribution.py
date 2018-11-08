import numpy as np

def index_coordinate_matrix(shape):
    """Generate a nd matrix of coordinate vectors"""
    ind = [slice(0,n) for n in shape]
    return np.mgrid[ind].transpose(np.roll(np.arange(len(shape)+1),-1))

def marginal(data, dim):
    """Return marginal of data, varying along dim (can be scalar or array of dims)"""
    dims = np.array([dim]).flatten()
    m = data
    for d in np.arange(data.ndim-1,0-1,-1):
        if d in dims:
            continue
        m = m.sum(axis=d)
    return m

def moments(data, num, dim=0, normalized=False,keepdims=False):
    """Compute up to the num-th moment of the distribution data along specified axis (default 0)"""
    # to make things easier, swap dim into position 0 here, then swap it back if keepDims is True
    if dim != 0:
        data = data.swapaxes(0,dim)
    if not normalized:
        data = data / data.sum(axis=0)[None,...]
    # create support vector along (0) dim
    xshape = np.ones(data.ndim)
    n = data.shape[0]
    xshape[0] = n
    x = np.arange(n).reshape(xshape.astype(int))
    # output shape is same as input shape, but size along dim 0 gets replaced with num_moments
    mshape = np.array(data.shape)
    mshape[0] = num
    m = np.zeros(mshape)
    # first moments are means and var
    if num > 0:
        m[0] = (data * x).sum(axis=0)
    if num > 1:
        m[1] = (data * (x - m[0])**2).sum(axis=0)
    # beyond var, return normalized moments
    for ii in range(2,num):
        m[ii] = (data * (x - m[0])**(ii+1)).sum(axis=0) / m[1]**(float((ii+1))/2)
    if keepdims:
        m = m.swapaxes(0,dim)
    return m
