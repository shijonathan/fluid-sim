import taichi as ti

ti.init(arch=ti.gpu, debug=True)

# 3-D MPM 8 particles per grid
n_particles = 8192
n_grid = 64
dx = 1 / n_grid # Grid spacing
dt = 2e-4 # Time step
dim = 3

# Physical parameters
bound = 3
p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
E, nu = 1000, 0.2 # Young's modulus, Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

# Field initialization
x = ti.Vector.field(dim, float, n_particles) # Position
v = ti.Vector.field(dim, float, n_particles) # Velocity
C = ti.Matrix.field(dim, dim, float, n_particles) # Affine transformation
F = ti.Matrix.field(dim, dim, float, n_particles) # Deformation gradient
Jp = ti.field(float, n_particles) # Total deformation

# Grid initialization
grid_v = ti.Vector.field(dim, float, (n_grid, n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid, n_grid))

used = ti.field(int, n_particles)

neighbor = (3, ) * dim

# Adding a force field to hit a target
# force_field = False
show_target = False
target_pos = ti.Vector.field(dim, float, 1)
target_pos[0] = ti.Vector([0.0, 0.5, 0.0])

line_pos = ti.Vector.field(dim, float, 2)
line_pos[0] = ti.Vector([0.0, 0.5, 0.0])
line_pos[1] = ti.Vector([0.96, 0.05, 0.96])

FORCE_STRENGTH = [0.0]

@ti.kernel
def substep(fs: float):
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.zero(grid_v[I])
        grid_m[I] = 0

    ti.loop_config(block_dim=n_grid)
    # Particle-to-grid transfer
    for p in x:
        if used[p] == 0:
            continue
        Xp = x[p] / dx
        base = int(Xp-0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2] # weights from mpm3d_ggui.py

        # Update particle deformation gradient
        F[p] = (ti.Matrix.identity(float, 3) + dt * C[p]) @ F[p]

        # Lame parameters
        la = lambda_0 * ti.exp(10 * (1.0 - Jp[p]))
        mu = 0 # for water

        # Singular value decomposition
        U, sig, V = ti.svd(F[p])
        J = 1.0

        for d in ti.static(range(dim)):
            new_sig = sig[d, d]
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        
        # Reset deformation gradient to avoid numerical instability
        new_F = ti.Matrix.identity(float, 3)
        new_F[0, 0] = J
        F[p] = new_F

        # Calculate stress/affine transformation
        stress = ti.Matrix.identity(float, 3) * la * J * (J - 1)
        stress = (-dt * p_vol * 4) * stress / dx**2
        affine = stress + p_mass * C[p]

        # Loop over 3x3 grid node neighborhood
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbor))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

    for I in ti.grouped(grid_m):
        # Computing grid velocities
        if grid_m[I] > 0:
            grid_v[I] /= grid_m[I] # Divide off mass to obtain velocity
        grid_v[I] += dt * ti.Vector([0, -9.8, 0])

        # Enforce boundary conditions
        cond = (I < bound) & (grid_v[I] < 0) | (I > n_grid - bound) & (grid_v[I] > 0) 
        grid_v[I] = ti.select(cond, 0, grid_v[I]) 

    # Grid-to-particle transfer
    ti.loop_config(block_dim=n_grid)
    for p in x:
        if used[p] == 0:
            continue
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.zero(v[p])
        new_C = ti.zero(C[p])

        for offset in ti.static(ti.grouped(ti.ndrange(*neighbor))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2

        # Force field
        new_v += fs * (ti.Vector([0.0, 0.5, 0.0]) - x[p])

        # Particle advection
        v[p] = new_v
        x[p] += dt * v[p]
        C[p] = new_C

class Cube:
    def __init__(self, minimum, size):
        self.minimum = minimum
        self.size = size
        self.volume = self.size.x * self.size.y * self.size.z

@ti.kernel
def init_cube_vol(first_par: int, last_par: int, x_begin: float, y_begin: float, z_begin: float, x_size: float, y_size: float, z_size: float):
    for i in range(first_par, last_par):
        x[i] = ti.Vector([ti.random() for i in range(dim)]) * ti.Vector(
            [x_size, y_size, z_size]) + ti.Vector([x_begin, y_begin, z_begin])
        Jp[i] = 1
        F[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        v[i] = ti.Vector([0.0, 0.0, 0.0])
        used[i] = 1

@ti.kernel
def set_all_unused():
    for p in used:
        used[p] = 0
        # basically throw them away so they aren't rendered
        x[p] = ti.Vector([533799.0, 533799.0, 533799.0])
        Jp[p] = 1
        F[p] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        C[p] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        v[p] = ti.Vector([0.0, 0.0, 0.0])

def init_vols(vols):
    set_all_unused()
    total_vol = 0
    for v in vols:
        total_vol += v.volume

    next_p = 0
    for i, v in enumerate(vols):
        v = vols[i]
        if isinstance(v, Cube):
            par_count = int(v.volume / total_vol * n_particles)
            if i == len(vols) - 1:  # this is the last volume, so use all remaining particles
                par_count = n_particles - next_p
            init_cube_vol(next_p, next_p + par_count, *v.minimum, *v.size)
            next_p += par_count
        else:
            raise Exception("???")

res = (1080, 720)
window = ti.ui.Window("MPM 3D", res, vsync=True)

canvas = window.get_canvas()
gui = window.get_gui()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.5, 0.3, 0.5)
camera.fov(55)

def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    
    scene.ambient_light((0, 0, 0))

    scene.particles(x, color=(0.1, 0.75, 0.9), radius=0.01)

    if show_target:
        scene.particles(target_pos, color=(1.0, 0.0, 0.0), radius=0.02)

    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))

    canvas.scene(scene)

preset = [[Cube(ti.Vector([0.55, 0.05, 0.55]), ti.Vector([0.4, 0.4, 0.4]))]]

def show_options():
    global show_target
    with gui.sub_window("Options", 0.05, 0.1, 0.3, 0.15) as w:
        show_target = w.checkbox("Show target", show_target)

        FORCE_STRENGTH[0] = w.slider_float("Force strength", FORCE_STRENGTH[0], 0, 0.1)

        if w.button("Restart"):
            init_vols(preset[0])

def main():
    init_vols(preset[0])
    while window.running:
        for _ in range(25):
            # if _ == 24:
            #     print(FORCE_STRENGTH)
            substep(*FORCE_STRENGTH)

        render()
        show_options()
        window.show()

if __name__ == '__main__':
    main()