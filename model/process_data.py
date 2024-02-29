import numpy as np
import scipy.io
import torch

# read in data
def read_data(f):
    # .mat files
    if '.mat' in f:
        data = scipy.io.loadmat(f)
        return data
    # TODO: handle other filetypes
    else:
        return f

# TODO: generalize, standardize what the input data looks like
# for now, just lid driven cavity problem data preprocessing
# U, X, t are flattened tensors
def preprocess(data):
    N_train = 5000

    usol = data['usol'] # sqrt(N) x sqrt(N) x T
    vsol = data['vsol'] # sqrt(N) x sqrt(N) x T
    psol = data['psol'] # sqrt(N) x sqrt(N) x T

    # cut off first entries from usol, vsol, and psol because it is empty
    usol = usol[:, :89, 1:] # cut off an extra row in y dimension
    vsol = vsol[:89, :, 1:] # cut off an extra row in x dimension 
    psol = psol[:89, :89, 1:]

    x = np.arange(0, 1, 1/90)[:89]
    y = np.arange(0, 1, 1/90)[:89]
    t = np.arange(0, 4, 0.01).reshape(-1, 1)[:399]

    X, Y = np.meshgrid(x, y)
    X_star = np.stack((X.ravel(), Y.ravel()), axis=1) # N x 2

    N = X_star.shape[0]
    T = t.shape[0]

    usol = np.reshape(usol, (usol.shape[0] * usol.shape[1], usol.shape[2])) # N x T
    vsol = np.reshape(vsol, (vsol.shape[0] * vsol.shape[1], vsol.shape[2])) # N x T
    psol = np.reshape(psol, (psol.shape[0] * psol.shape[1], psol.shape[2])) # N x T

    XX = np.tile(X_star[:, 0:1], (1, usol.shape[1])) # N x T
    YY = np.tile(X_star[:, 1:2], (1, usol.shape[1])) # N x T
    TT = np.tile(t, (1, X_star.shape[0])).T # N x T

    # flatten all inputs
    x = XX.flatten()[:, None] # NT x 1
    y = YY.flatten()[:, None] # NT x 1
    t = TT.flatten()[:, None] # NT x 1

    u = usol.flatten()[:, None] # NT x 1
    v = vsol.flatten()[:, None] # NT x 1
    p = psol.flatten()[:, None] # NT x 1

    idx = np.random.choice(usol.shape[0] * usol.shape[1], N_train, replace=False)
    x_train = x[idx, :]
    y_train = y[idx, :]
    t_train = t[idx, :]
    u_train = u[idx, :]
    v_train = v[idx, :]
    p_train = p[idx, :]