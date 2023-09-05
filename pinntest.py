import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import scipy.io

class PINN(nn.Module):
    def __init__(self, x, y, t, u, v, layers):
        super(PINN, self).__init__()

        X = np.concatenate([x, y, t], axis=1)

        self.lb = torch.Tensor(X.min(0))
        self.ub = torch.Tensor(X.max(0))

        self.X = torch.Tensor(X)

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.t = torch.Tensor(t)
        self.u = torch.Tensor(u)
        self.v = torch.Tensor(v)

        self.layers = layers

        self.weights, self.biases = self.initialize(layers)

        self.lambda_1 = nn.Parameter(torch.Tensor([0.0]))
        self.lambda_2 = nn.Parameter(torch.Tensor([0.0]))

        self.x_torch = None
        self.y_torch = None
        self.t_torch = None

        self.u_torch = None
        self.v_torch = None

        self.u_pred = None
        self.v_pred = None
        self.p_pred = None
        self.f_u_pred = None
        self.f_v_pred = None

        self.loss = None
        self.optimizer = None
        self.optimizer_Adam = None
        self.train_op_Adam = None

        self.initialize_model()


    def initialize(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = nn.Parameter(torch.zeros(1, layers[l+1]))
            weights.append(W)
            biases.append(b)

        return weights, biases
    
    def xavier_init(self, size):
        xavier_stddev = np.sqrt(2 / (size[0] + size[1]))
        return nn.Parameter(torch.randn(size[0], size[1]) * xavier_stddev)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(num_layers-2):
            W = weights[l]
            b = biases[l]
            H = torch.tanh(torch.add(torch.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = torch.add(torch.matmul(H, W), b)
        return Y

    def net_NS(self, x, y, t):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2

        psi_and_p = self.neural_net(torch.cat([x, y, t], dim=1), self.weights, self.biases)
        psi = psi_and_p[:, 0:1]
        p = psi_and_p[:, 1:2]

        u = torch.autograd.grad(psi, y, torch.ones_like(psi), create_graph=True)[0]
        v = -torch.autograd.grad(psi, x, torch.ones_like(psi), create_graph=True)[0]

        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]

        v_t = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]

        p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True)[0]

        f_u = u_t + lambda_1 * (u * u_x + v * u_y) + p_x - lambda_2 * (u_xx + u_yy)
        f_v = v_t + lambda_1 * (u * v_x + v * v_y) + p_y - lambda_2 * (v_xx + v_yy)

        return u, v, p, f_u, f_v

    def callback(self, loss, lambda_1, lambda_2):
        print('Loss: %.3e, l1: %.3f, l2: %.5f' % (loss, lambda_1, lambda_2))


    def initialize_model(self):
        self.x_torch = torch.nn.Parameter(torch.zeros(1, 1), requires_grad=False)
        self.y_torch = torch.nn.Parameter(torch.zeros(1, 1), requires_grad=False)
        self.t_torch = torch.nn.Parameter(torch.zeros(1, 1), requires_grad=False)
        self.u_torch = torch.nn.Parameter(torch.zeros(1, 1), requires_grad=False)
        self.v_torch = torch.nn.Parameter(torch.zeros(1, 1), requires_grad=False)

        self.u_pred, self.v_pred, self.p_pred, self.f_u_pred, self.f_v_pred = self.net_NS(self.x_torch, self.y_torch, self.t_torch)

        self.loss = torch.sum(torch.square(self.u_torch - self.u_pred)) + \
                    torch.sum(torch.square(self.v_torch - self.v_pred)) + \
                    torch.sum(torch.square(self.f_u_pred)) + \
                    torch.sum(torch.square(self.f_v_pred))

        self.optimizer_Adam = optim.Adam(self.parameters(), lr=0.001)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

    def train(self, nIter):
        torch_dict = {self.x_torch: self.x, self.y_torch: self.y, self.t_torch: self.t,
                   self.u_torch: self.u, self.v_torch: self.v}

        start_time = time.time()
        for it in range(nIter):
            self.optimizer_Adam.zero_grad()
            self.train_op_Adam.zero_grad()
            
            self.u_pred, self.v_pred, self.p_pred, self.f_u_pred, self.f_v_pred = self.net_NS(self.x_torch, self.y_torch, self.t_torch)
            
            loss_value = self.loss
            loss_value.backward()
            
            self.optimizer_Adam.step()

            if it % 10 == 0:
                elapsed = time.time() - start_time
                lambda_1_value = self.lambda_1.item()
                lambda_2_value = self.lambda_2.item()
                print('It: %d, Loss: %.3e, l1: %.3f, l2: %.5f, Time: %.2f' % 
                      (it, loss_value.item(), lambda_1_value, lambda_2_value, elapsed))
                start_time = time.time()    

if __name__ == "__main__":
    nIter = 5000
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]

    data = scipy.io.loadmat(r"C:\Users\jshi3\Documents\\fluid-sim\data\cylinder_nektar_wake.mat")

    U_star = data['U_star'] # N x 2 x T
    P_star = data['p_star'] # N x T
    t_star = data['t'] # T x 1
    X_star = data['X_star'] # N x 2
    
    N = X_star.shape[0]
    T = t_star.shape[0]
    
    # Rearrange Data 
    XX = np.tile(X_star[:,0:1], (1,T)) # N x T
    YY = np.tile(X_star[:,1:2], (1,T)) # N x T
    TT = np.tile(t_star, (1,N)).T # N x T
    
    UU = U_star[:,0,:] # N x T
    VV = U_star[:,1,:] # N x T
    PP = P_star # N x T
    
    x = XX.flatten()[:,None] # NT x 1
    y = YY.flatten()[:,None] # NT x 1
    t = TT.flatten()[:,None] # NT x 1
    
    u = UU.flatten()[:,None] # NT x 1
    v = VV.flatten()[:,None] # NT x 1
    p = PP.flatten()[:,None] # NT x 1
  
    # Training Data    
    idx = np.random.choice(N*T, nIter, replace=False)
    x_train = x[idx,:]
    y_train = y[idx,:]
    t_train = t[idx,:]
    u_train = u[idx,:]
    v_train = v[idx,:]

    # Training
    model = PINN(x_train, y_train, t_train, u_train, v_train, layers)
    model.train(nIter)  # Train the model

    snap = np.array([100])
    x_star = X_star[:,0:1]
    y_star = X_star[:,1:2]
    t_star = TT[:,snap]
    
    u_star = U_star[:,0,snap]
    v_star = U_star[:,1,snap]
    p_star = P_star[:,snap]
    
    # Prediction
    u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)
    lambda_1_value = model.sess.run(model.lambda_1)
    lambda_2_value = model.sess.run(model.lambda_2)
    
    # Error
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_p = np.linalg.norm(p_star-p_pred,2)/np.linalg.norm(p_star,2)

    error_lambda_1 = np.abs(lambda_1_value - 1.0)*100
    error_lambda_2 = np.abs(lambda_2_value - 0.01)/0.01 * 100
    
    print('Error u: %e' % (error_u))    
    print('Error v: %e' % (error_v))    
    print('Error p: %e' % (error_p))    
    print('Error l1: %.5f%%' % (error_lambda_1))                             
    print('Error l2: %.5f%%' % (error_lambda_2))