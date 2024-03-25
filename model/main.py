from PINN import *
from process_data import *
from generate_plots import *

def main():
    filename = 'C:\\Users\\Administrator\\Documents\\research\\fluid-sim\\data\\2d_navierstokes.mat'
    data = read_data(filename)
    x, y, t, u, v = preprocess(data)

    PINN = NS(x, y, t, u, v)
    PINN.train()
    PINN.net.eval()

if __name__ == "__main__":
    main()