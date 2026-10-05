import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# print(torch.cuda.is_available())
torch.manual_seed(42)
np.random.seed(42)

# device = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )
# device
# Force CPU mapping for testing
device = torch.device("cpu") 
# checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# Physical parameters                                
L = 2.0
H = 1.0
T = 1.0
rho = 1000.0
nu = 1e-3                                   
g = 9.81

# opening on the right wall
opening_y_min = 0.30
opening_y_max = 0.70

# Interior points
N_f = 10000
x_f = np.random.uniform(0, L, N_f)
y_f = np.random.uniform(0, H, N_f)
t_f = np.random.uniform(0, T, N_f)

X_f = np.column_stack([x_f, y_f, t_f])
X_f = torch.tensor(
    X_f,
    dtype=torch.float32,
    device=device,
    requires_grad=True
)

print(X_f.shape)

# Generate wall points 

# Bottom boundary (x in [0, L], y=0, t in [0, T])
N_wall = 1000
x_bottom = np.linspace(0, L, N_wall)
bottom_pts = np.column_stack([
    x_bottom,
    np.zeros(N_wall),
    np.linspace(0, T, N_wall)
])

# Left boundary (x=0, y in [0, H], t in [0, T])
y_left = np.linspace(0, H, N_wall)
left_pts = np.column_stack([
    np.zeros(N_wall),
    y_left,
    np.linspace(0, T, N_wall)
])

# Right side wall (0->0.30, 0.70->1)
y_right_bottom_side = np.linspace(0, opening_y_min, N_wall // 2)
y_right_top_side = np.linspace(opening_y_max, H, N_wall // 2)

# (x=L, y=0->0.30, t in [0, T])
right_bottom_pts = np.column_stack([
    np.full(len(y_right_bottom_side), L),
    y_right_bottom_side,
    np.linspace(0, T, len(y_right_bottom_side))
])

# (x=L, y=0.70->1, t in [0, T])
right_top_pts = np.column_stack([
    np.full(len(y_right_top_side), L),
    y_right_top_side,
    np.linspace(0, T, len(y_right_top_side))
])

# Combining all wall points 
X_wall = np.vstack([left_pts, bottom_pts, right_bottom_pts, right_top_pts])
X_wall = torch.tensor(X_wall, dtype=torch.float32, device=device)
print(X_wall.shape)

# Generating opening points
N_open = 500
y_open = np.linspace(opening_y_min, opening_y_max, N_open)
t_open = np.linspace(0, T, N_open)
opening_pts = np.column_stack([
    np.full(len(y_open), L),
    y_open,
    t_open
])
X_open = torch.tensor(opening_pts, dtype=torch.float32, device=device)
X_open.shape

class PINN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.network(x)

model = PINN().to(device)
test_pnts = torch.tensor([[0.2, 1.2, 0.5]], dtype=torch.float32, device=device)

pred = model(test_pnts)
print(pred)

print(pred[0,0].item())

def derivatives(output, inputs):
    grad = torch.autograd.grad(
        output,
        inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True
    )[0]
    
    return grad

# continuity loss calculation
def continuity_loss(model, x, y, t):
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    X = torch.stack((x, y, t), dim=1)
    
    pred = model(X)
    
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    # p = pred[:, 2:3]
    
    u_x = derivatives(u, x)
    v_y = derivatives(v, y)
    
    R_cont = u_x + v_y
    
    return R_cont

# x-momentum loss
def x_momentum_loss(model, x, y, t):
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    X = torch.stack((x, y, t), dim=1)
    
    pred = model(X)
    
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    p = pred[:, 2:3]
    
    u_x = derivatives(u, x)
    u_y = derivatives(u, y)
    
    u_xx = derivatives(u_x, x)
    u_yy = derivatives(u_y, y)
    
    p_x = derivatives(p, x)

    u_t = derivatives(u, t)
    
    R_x = u_t + u*u_x + v*u_y + ((1/rho)*p_x) - nu*(u_xx + u_yy)
    
    return R_x

# y-momentum loss
# NOTE: the network now predicts DYNAMIC pressure only.
# True pressure p = p_dynamic + p_hydrostatic, where p_hydrostatic = -rho*g*y
# so d(p_hydrostatic)/dy = -rho*g exactly cancels the gravity term below.
# This keeps the network's pressure output O(1) instead of needing to
# represent a ~-9810 Pa/m gradient, which was stalling the physics loss.
def y_momentum_loss(model, x, y, t):
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    X = torch.stack((x, y, t), dim=1)
    
    pred = model(X)
    
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    p_dynamic = pred[:, 2:3]
    
    v_x = derivatives(v, x)
    v_y = derivatives(v, y)
    
    v_xx = derivatives(v_x, x)
    v_yy = derivatives(v_y, y)

    v_t = derivatives(v, t)
    
    p_y = derivatives(p_dynamic, y)
    
    R_y = v_t + u*v_x + v*v_y + (1/rho)*p_y - nu*(v_xx + v_yy)

    return R_y

print(X_f[:, 0], X_f[:, 1], X_f[:, 2])

R_cont  = continuity_loss(model, X_f[:, 0], X_f[:, 1], X_f[:, 2])
R_x     = x_momentum_loss(model, X_f[:, 0], X_f[:, 1], X_f[:, 2])
R_y     = y_momentum_loss(model, X_f[:, 0], X_f[:, 1], X_f[:, 2])

print("Continuity residual:",
      R_cont.abs().mean().item())

print("X momentum residual:",
      R_x.abs().mean().item())

print("Y momentum residual:",
      R_y.abs().mean().item())

def physics_loss():
    x = X_f[:, 0]
    y = X_f[:, 1]
    t = X_f[:, 2]

    R_cont = continuity_loss(model, x, y, t)
    R_x = x_momentum_loss(model, x, y, t)
    R_y = y_momentum_loss(model, x, y, t)

    loss_cont = torch.mean(R_cont**2)
    loss_x = torch.mean(R_x**2)
    loss_y = torch.mean(R_y**2)

    return loss_cont + loss_x + loss_y

# Wall boundary loss (u=0, v=0)
def wall_loss():

    prediction = model(X_wall)

    u_wall = prediction[:, 0]
    v_wall = prediction[:, 1]

    loss_u = torch.mean(u_wall**2)
    loss_v = torch.mean(v_wall**2)

    return loss_u + loss_v

# Opening boundary condition (p=0)
def opening_loss():
    
    prediction = model(X_open)
    p_open = prediction[:, 2]

    return torch.mean(p_open**2)

X_pressure_ref = torch.tensor(
    [[0.0, H, 0.0]],
    dtype=torch.float32,
    device=device
)

def pressure_reference_loss():

    prediction = model(X_pressure_ref)
    p_ref = prediction[:, 2]

    return torch.mean(p_ref**2)

# Initial condition points (t = 0, u = v = 0 everywhere in the domain)
N_ic = 2000
x_ic = np.random.uniform(0, L, N_ic)
y_ic = np.random.uniform(0, H, N_ic)
t_ic = np.zeros(N_ic)

X_ic = np.column_stack([x_ic, y_ic, t_ic])
X_ic = torch.tensor(X_ic, dtype=torch.float32, device=device)
print(X_ic.shape)

def ic_loss():
    pred = model(X_ic)
    u_ic = pred[:, 0]
    v_ic = pred[:, 1]
    return torch.mean(u_ic**2) + torch.mean(v_ic**2)

# Total loss
# Loss weights -- tune these; wall/opening usually need to be weighted up
# relative to physics because rho=1000 makes physics-loss gradients dominate
lambda_phys = 1.0
lambda_wall = 10.0
lambda_open = 10.0
lambda_ref  = 1.0
lambda_ic   = 10.0

def total_loss():

    L_phys = physics_loss()
    L_wall = wall_loss()
    L_open = opening_loss()
    L_ref  = pressure_reference_loss()
    L_ic   = ic_loss()

    loss = (
        lambda_phys * L_phys +
        lambda_wall * L_wall +
        lambda_open * L_open +
        lambda_ref  * L_ref  +
        lambda_ic   * L_ic
    )

    return loss, L_phys, L_wall, L_open, L_ref, L_ic

# --- LES turbulence closure parameters ---
# Used to build the effective viscosity nu_eff = nu + nu_t appearing in the
# full momentum equation (Eq. 2 in the reference image).
Cs = 0.1                                   # Smagorinsky constant (typical 0.1-0.2)
Delta_les = ((L * H) ** 0.5) / 20.0        # characteristic filter width (domain-scale / resolution)

# Smagorinsky-type LES eddy viscosity: nu_t = (Cs * Delta)^2 * |S|,  |S| = sqrt(2 S_ij S_ij)
def eddy_viscosity(u_x, u_y, v_x, v_y):
    S11 = u_x
    S22 = v_y
    S12 = 0.5 * (u_y + v_x)

    S_mag = torch.sqrt(2.0 * (S11**2 + S22**2 + 2.0 * S12**2) + 1e-12)
    nu_t = (Cs * Delta_les) ** 2 * S_mag

    return nu_t

# Complete x-momentum loss (full stress tensor + turbulence + gravity + surface tension)
def x_momentum_loss_full(model, x, y, t):
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    X = torch.stack((x, y, t), dim=1)

    pred = model(X)

    u = pred[:, 0:1]
    v = pred[:, 1:2]
    p = pred[:, 2:3]

    u_x = derivatives(u, x)
    u_y = derivatives(u, y)
    v_x = derivatives(v, x)
    v_y = derivatives(v, y)

    u_t = derivatives(u, t)
    p_x = derivatives(p, x)

    # effective viscosity nu_eff = nu (molecular) + nu_t (LES eddy viscosity)
    nu_t = eddy_viscosity(u_x, u_y, v_x, v_y)
    nu_eff = nu + nu_t

    # full symmetric viscous stress tensor components: tau_ij = nu_eff*(du_i/dx_j + du_j/dx_i)
    tau_xx = nu_eff * (u_x + u_x)
    tau_xy = nu_eff * (u_y + v_x)

    diffusion_x = derivatives(tau_xx, x) + derivatives(tau_xy, y)

    # gravity: g_x = 0 (gravity acts only in -y for this vertical tank)
    g_x = 0.0

    # surface tension: sigma*k*dphi/dx = 0 (single-phase domain, no free surface here)
    surface_tension_x = 0.0

    R_x = u_t + u*u_x + v*u_y + ((1/rho)*p_x) - diffusion_x - g_x - surface_tension_x

    return R_x

# Complete y-momentum loss (full stress tensor + turbulence + gravity + surface tension)
# NOTE: the network still predicts DYNAMIC pressure only (p = p_dynamic - rho*g*y),
# so gravity g_y is already supplied analytically by that decomposition and must NOT
# be added again here, or it would be double counted.
def y_momentum_loss_full(model, x, y, t):
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    X = torch.stack((x, y, t), dim=1)

    pred = model(X)

    u = pred[:, 0:1]
    v = pred[:, 1:2]
    p_dynamic = pred[:, 2:3]

    u_x = derivatives(u, x)
    u_y = derivatives(u, y)
    v_x = derivatives(v, x)
    v_y = derivatives(v, y)

    v_t = derivatives(v, t)
    p_y = derivatives(p_dynamic, y)

    # effective viscosity nu_eff = nu (molecular) + nu_t (LES eddy viscosity)
    nu_t = eddy_viscosity(u_x, u_y, v_x, v_y)
    nu_eff = nu + nu_t

    # full symmetric viscous stress tensor components: tau_ij = nu_eff*(du_i/dx_j + du_j/dx_i)
    tau_yx = nu_eff * (v_x + u_y)
    tau_yy = nu_eff * (v_y + v_y)

    diffusion_y = derivatives(tau_yx, x) + derivatives(tau_yy, y)

    # gravity: g_y already included analytically via the dynamic/hydrostatic pressure split
    # (see note above) -> do not add again here.

    # surface tension: sigma*k*dphi/dy = 0 (single-phase domain, no free surface here)
    surface_tension_y = 0.0

    R_y = v_t + u*v_x + v*v_y + (1/rho)*p_y - diffusion_y - surface_tension_y

    return R_y

# Complete physics loss: mass conservation (unchanged) + full momentum equations
def physics_loss_full():
    x = X_f[:, 0]
    y = X_f[:, 1]
    t = X_f[:, 2]

    R_cont = continuity_loss(model, x, y, t)          # mass conservation (Eq. 1, unchanged)
    R_x    = x_momentum_loss_full(model, x, y, t)      # complete x-momentum (Eq. 2)
    R_y    = y_momentum_loss_full(model, x, y, t)      # complete y-momentum (Eq. 2)

    loss_cont = torch.mean(R_cont**2)
    loss_x    = torch.mean(R_x**2)
    loss_y    = torch.mean(R_y**2)

    return loss_cont + loss_x + loss_y

# Complete total loss (physics now uses the full Navier-Stokes momentum equation)
def total_loss_full():

    L_phys = physics_loss_full()
    L_wall = wall_loss()
    L_open = opening_loss()
    L_ref  = pressure_reference_loss()
    L_ic   = ic_loss()

    loss = (
        lambda_phys * L_phys +
        lambda_wall * L_wall +
        lambda_open * L_open +
        lambda_ref  * L_ref  +
        lambda_ic   * L_ic
    )

    return loss, L_phys, L_wall, L_open, L_ref, L_ic

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=200
)

# epochs = 1000
# resample_every = 200   # refresh collocation points periodically
# loss_history = []
# loss_components_history = []

# def resample_collocation_points():
#     global X_f
#     x_f = np.random.uniform(0, L, N_f)
#     y_f = np.random.uniform(0, H, N_f)
#     t_f = np.random.uniform(0, T, N_f)
#     X_new = np.column_stack([x_f, y_f, t_f])
#     X_f = torch.tensor(X_new, dtype=torch.float32, device=device, requires_grad=True)

# # --- Phase 1: Adam ---
# for epoch in range(epochs):

#     if epoch % resample_every == 0 and epoch > 0:
#         resample_collocation_points()

#     optimizer.zero_grad()
#     (loss, L_phys, L_wall, L_open, L_ref, L_ic) = total_loss_full()
#     loss.backward()
#     optimizer.step()
#     scheduler.step(loss)

#     loss_history.append(loss.item())
#     loss_components_history.append(
#         (L_phys.item(), L_wall.item(), L_open.item(), L_ref.item(), L_ic.item())
#     )

#     if epoch % 5 == 0:
#         current_lr = optimizer.param_groups[0]['lr']
#         print(
#             f"Epoch {epoch+1:5d} | "
#             f"Total: {loss.item():.6f} | "
#             f"Physics: {L_phys.item():.6f} | "
#             f"Wall: {L_wall.item():.6f} | "
#             f"Open: {L_open.item():.6f} | "
#             f"Ref: {L_ref.item():.6f} | "
#             f"IC: {L_ic.item():.6f} | "
#             f"LR: {current_lr:.2e}"
#         )

# # --- Phase 2: L-BFGS fine-tuning ---
# lbfgs = torch.optim.LBFGS(
#     model.parameters(), lr=1.0, max_iter=500,
#     history_size=50, line_search_fn="strong_wolfe"
# )

# lbfgs_iter = [0]
# def closure():
#     lbfgs.zero_grad()
#     loss, L_phys, L_wall, L_open, L_ref, L_ic = total_loss_full()
#     loss.backward()
#     loss_history.append(loss.item())
#     loss_components_history.append(
#         (L_phys.item(), L_wall.item(), L_open.item(), L_ref.item(), L_ic.item())
#     )
#     if lbfgs_iter[0] % 20 == 0:
#         print(f"L-BFGS iter {lbfgs_iter[0]:4d} | Total: {loss.item():.6f}")
#     lbfgs_iter[0] += 1
#     return loss

# lbfgs.step(closure)
# print("L-BFGS fine-tuning complete.")

# # --- Save checkpoint ---
# checkpoint_path = "pinn_model_with_eddy.pt"
# checkpoint = {
#     "epoch": epochs,
#     "model_state_dict": model.state_dict(),
#     "optimizer_state_dict": optimizer.state_dict(),
#     "loss_history": loss_history,
#     "loss_components_history": loss_components_history,
# }
# torch.save(checkpoint, checkpoint_path)
# print(f"Training complete. Checkpoint saved to {checkpoint_path}")

import torch
import torch.nn as nn

# Make sure this matches the model definition used during training
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.network(x)

# Recreate the model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Load checkpoint
checkpoint_path = r"C:\Users\ps302\OneDrive\Desktop\PINN\src\checkpoints\pinn_model_with_eddy.pt"
# checkpoint = torch.load(checkpoint_path, map_location=device)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

# Restore model weights
model.load_state_dict(checkpoint["model_state_dict"])

# Restore optimizer state if needed
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# Restore extra metadata
epoch = checkpoint["epoch"]
loss_history = checkpoint["loss_history"]
loss_components_history = checkpoint.get("loss_components_history", [])

print(f"Loaded checkpoint from {checkpoint_path}")
print(f"Last epoch: {epoch}")
print(f"Loss history length: {len(loss_history)}")

plt.figure(figsize=(7, 5))

plt.semilogy(
    loss_history
)

plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.title("PINN Training Convergence")

plt.grid(True)
plt.show()

components = np.array(loss_components_history)
labels = ["Physics", "Wall", "Opening", "Pressure Ref", "IC"]

plt.figure(figsize=(8, 5))
for i, label in enumerate(labels):
    plt.semilogy(components[:, i], label=label)
plt.xlabel("Epoch")
plt.ylabel("Loss (log scale)")
plt.title("Individual Loss Components")
plt.legend()
plt.grid(True)
plt.show()

Nx = 100
Ny = 100
t_plot = 0.5

x = np.linspace(0, L, Nx)
y = np.linspace(0, H, Ny)

X, Y = np.meshgrid(x, y)

XY = np.column_stack([
    X.flatten(),
    Y.flatten(),
    np.full(X.size, t_plot)
])

XY_tensor = torch.tensor(
    XY,
    dtype=torch.float32,
    device=device
)

with torch.no_grad():

    prediction = model(XY_tensor)

u_pred = prediction[:, 0].cpu().numpy()
v_pred = prediction[:, 1].cpu().numpy()
p_dynamic_pred = prediction[:, 2].cpu().numpy()

u_pred = u_pred.reshape(Ny, Nx)
v_pred = v_pred.reshape(Ny, Nx)
p_dynamic_pred = p_dynamic_pred.reshape(Ny, Nx)

# Reconstruct TRUE pressure: p = p_dynamic + p_hydrostatic, p_hydrostatic = -rho*g*y
p_hydrostatic = -rho * g * Y
p_pred = p_dynamic_pred + p_hydrostatic


import numpy as np
import torch
import matplotlib.pyplot as plt


N_open = 500
x = np.linspace(0, 2, N_open)
y = np.linspace(0, 1, N_open)
t_plot = 0.5

X, Y = np.meshgrid(x, y)

x_flat = X.flatten()
y_flat = Y.flatten()
t_flat = np.full_like(x_flat, t_plot)

grid_pts = np.column_stack([x_flat, y_flat, t_flat])
X_grid_tensor = torch.tensor(grid_pts, dtype=torch.float32, device=device)

with torch.no_grad():
    grid_prediction = model(X_grid_tensor)

u_open = grid_prediction[:, 0].cpu().numpy().reshape(X.shape)
v_open = grid_prediction[:, 1].cpu().numpy().reshape(Y.shape)

# Velocity magnitude: |V| = sqrt(u^2 + v^2)
velocity_magnitude = np.hypot(u_open, v_open)

plt.figure(figsize=(8, 5))

stream = plt.streamplot(
    X,
    Y,
    u_open,
    v_open,
    color=velocity_magnitude,
    cmap="viridis",
    density=1.5,
    linewidth=1.5,
)

plt.colorbar(stream.lines, label="Velocity magnitude")

plt.xlabel("x")
plt.ylabel("y")
plt.title(f"PINN Fluid Flow at t = {t_plot}")
plt.show()