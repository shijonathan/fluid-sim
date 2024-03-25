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

    X, Y, T = np.meshgrid(x, y, t, indexing='ij')
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    T = T.reshape(-1, 1)
    X_star = np.concatenate((X, Y, T), axis=1).reshape(89, 89, 399, 3)

    N = X_star.shape[0] * X_star.shape[1]
    T = t.shape[0]

    x_bc = np.concatenate((X_star[:, 0, :], X_star[:, -1, :], X_star[0, 1:-1, :], X_star[-1, 1:-1, :]), axis=0)
    x_ic = X_star[:, :, 0].reshape(N, 3) # positions
    x_bc = x_bc.reshape(x_bc.shape[0] * x_bc.shape[1], x_bc.shape[2])
    x_icbc = np.vstack((x_ic, x_bc))

    # usol = np.reshape(usol, (N, T)) # N x T
    # vsol = np.reshape(vsol, (N, T)) # N x T
    # psol = np.reshape(psol, (N, T)) # N x T

    # XX = np.tile(X_star[:, 0:1], (1, usol.shape[1])) # N x T
    # YY = np.tile(X_star[:, 1:2], (1, usol.shape[1])) # N x T
    # TT = np.tile(t, (1, X_star.shape[0])).T # N x T

    # flatten all inputs
    # x = XX.flatten()[:, None] # NT x 1
    # y = YY.flatten()[:, None] # NT x 1
    # t = TT.flatten()[:, None] # NT x 1

    # u = usol.flatten()[:, None] # NT x 1
    # v = vsol.flatten()[:, None] # NT x 1
    # p = psol.flatten()[:, None] # NT x 1

    # extract initial and boundary conditions
    u_ic = usol[:, :, 0]
    u_bc = np.concatenate((usol[:, 0, :], usol[:, -1, :], usol[0, 1:-1, :], usol[-1, 1:-1, :]), axis=0)
    u_icbc = np.hstack((u_ic.flatten(), u_bc.flatten())).reshape(-1, 1)

    v_ic = vsol[:, :, 0]
    v_bc = np.concatenate((vsol[:, 0, :], vsol[:, -1, :], vsol[0, 1:-1, :], vsol[-1, 1:-1, :]), axis=0)
    v_icbc = np.hstack((v_ic.flatten(), v_bc.flatten())).reshape(-1, 1)

    idx = np.random.choice(u_icbc.shape[0], N_train, replace=False)
    x_train = x_icbc[idx][:, 0]
    y_train = x_icbc[idx][:, 1]
    t_train = x_icbc[idx][:, 2]

    u_train = u_icbc[idx, :]
    v_train = v_icbc[idx, :]
    # p_train = p[idx, :]

    return x_train, y_train, t_train, u_train, v_train