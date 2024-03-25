import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize deep neural network
class DNN(nn.Module):
    def __init__(self):  
        super(DNN, self).__init__()
        # Layers of the net
        self.layers = [3, 20, 20, 20, 20, 20, 20, 2]
        self.depth = len(self.layers) - 1

        # Activation function
        self.activation = nn.Tanh

        # Compiling network
        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(self.layers[i], self.layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(('layer_%d' % (self.depth - 1), nn.Linear(self.layers[-2], self.layers[-1])))

        layerDict = OrderedDict(layer_list)
        self.model = nn.Sequential(layerDict)
        self.model.apply(self.init_weights)

    # Xavier initilization of weights
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    # Forward pass
    def forward(self, x):
        out = self.model(x)
        return out
    
# Initialize PINN 
class NS:
    def __init__(self, X, Y, T, u, v, nu):
        self.x = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)
        self.y = torch.tensor(Y, dtype=torch.float32, requires_grad=True).to(device)
        self.t = torch.tensor(T, dtype=torch.float32, requires_grad=True).to(device)

        self.u = torch.tensor(u, dtype=torch.float32).to(device)
        self.v = torch.tensor(v, dtype=torch.float32).to(device)

        self.nu = nu

        self.null = torch.zeros((self.x.shape[0], 1)).to(device)
        
        self.net = DNN().to(device)

        # Track loss history 
        self.loss_history = []
        self.loss_history_u = []
        self.loss_history_v = []
        self.loss_history_f = []
        self.loss_history_g = []

        # Optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.1)

        # self.optimizer = torch.optim.LBFGS(self.net.parameters(), lr=1, max_iter=200000, max_eval=50000, 
        #                                    history_size=50, tolerance_grad=1e-5, tolerance_change=0.5 * np.finfo(float).eps, line_search_fn="strong_wolfe")
        
        # Mean squared loss
        self.mse = nn.MSELoss().to(device)

        self.ls = 0
        self.iter = 0

    def function(self, x, y, t):
        res = self.net(torch.hstack((x, y, t)))
        psi, p = res[:, 0:1], res[:, 1:2]

        u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), 
                                create_graph=True)[0]
        v = -1.*torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), 
                                    create_graph=True)[0]

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), 
                                  create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), 
                                   create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), 
                                  create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), 
                                   create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), 
                                  create_graph=True)[0]

        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), 
                                  create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), 
                                   create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), 
                                  create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), 
                                   create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), 
                                  create_graph=True)[0]

        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), 
                                  create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), 
                                  create_graph=True)[0]

        f = u_t + u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        g = v_t + u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)

        return u, v, p, f, g
    
    def closure(self):
        self.optimizer.zero_grad()

        u_pred, v_pred, p_pred, f_pred, g_pred = self.function(self.x, self.y, self.t)

        u_loss = self.mse(u_pred, self.u)
        v_loss = self.mse(v_pred, self.v)
        f_loss = self.mse(f_pred, self.null)
        g_loss = self.mse(g_pred, self.null)

        self.ls = u_loss + v_loss + f_loss + g_loss

        self.loss_history.append(self.ls.detach().cpu().numpy())
        self.loss_history_u.append(u_loss.detach().cpu().numpy())
        self.loss_history_v.append(v_loss.detach().cpu().numpy())
        self.loss_history_f.append(f_loss.detach().cpu().numpy())
        self.loss_history_g.append(g_loss.detach().cpu().numpy())

        self.ls.backward()

        self.iter += 1
        if not self.iter % 1: 
            print('Iteration: {:}, Loss: {:0.6f}'.format(self.iter, self.ls))

        return self.ls
    
    def train(self):
        # Adam training
        while self.iter < 10:
            self.net.train()
            self.optimizer.step(self.closure)
        
        # L-BFGS training
        if self.iter == 10:
            print('--------Switching optimizer--------')
            print(self.optimizer)
            self.optimizer = torch.optim.LBFGS(self.net.parameters(), lr=1, max_iter=200000, max_eval=50000,history_size=50, tolerance_grad=1e-5, tolerance_change=0.5 * np.finfo(float).eps, line_search_fn="strong_wolfe")
            print(self.optimizer)
            print('-----------------------------------')
        self.net.train()
        self.optimizer.step(self.closure)