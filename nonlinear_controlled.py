
# We are interested in finding a control of the nonlinear system (1.7):
# \begin{equation}
# 		\left\{
# 		\begin{aligned}
# 			&h(t)\psi_{t}-\psi_{xx} - \frac{x}{2}h^{\prime}(t)\psi_{x}-h(t)\psi(1-\psi) =0 & & \text {in}\; Q, \\
# 			& \psi_{x}(0,t)=u(t)  & & \text {in}\;(0,T), \\
# 			& \psi(1,t)=0 & & \text {in}\;(0,T), \\
# 			&h^{\prime}(t)+2\mu \psi_{x}(1,t)=0 & & \text {in}\;(0,T),  \\
# 			& h(0)=h_{0}\\
# 			& \psi(\cdot,0)=\psi_{0} & &\text {in}\;(0,1),
# 		\end{aligned}
# 		\right.
# 	\end{equation}
# 	such that
# 	$$\psi(x,T)=0, ∀ x\in (0,1), \;\; and\;\; h(T)=1$$
# We consider the following data:
# 
# $L=1$, $T=1$, $h_0=0.5$, $\psi_0(x)=\sin(\pi x)$ for $0\leq x\leq L$. 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
### Parameters

mu = 0.5  # Coefficient in the dynamic condition
L = 1    # Spatial domain (-1, 1)
T = 1   # Time domain (0, T)

delta = 1e-6 # to avoid singularities

### Networks for the state and control variables

# Network for the state z(x,t)
psi_net = tf.keras.Sequential([
    tf.keras.layers.InputLayer((2,)),
    tf.keras.layers.Dense(50, activation='tanh'),
    tf.keras.layers.Dense(50, activation='tanh'),
    tf.keras.layers.Dense(1)
])

# Network for the state k(t)
h_net = tf.keras.Sequential([
    tf.keras.layers.InputLayer((1,)),
    tf.keras.layers.Dense(50, activation='tanh'),
    tf.keras.layers.Dense(50, activation='tanh'),
    tf.keras.layers.Dense(1)
])

# Network for the control 
u_net = tf.keras.Sequential([
    tf.keras.layers.InputLayer((2,)),
    tf.keras.layers.Dense(50, activation='tanh'),
    tf.keras.layers.Dense(50, activation='tanh'),
    tf.keras.layers.Dense(1)
])

### Dataset for training and test (Training and test points)

# training: random points that follow a uniform distribution
N_x_training = 1000
N_t_training = 1000

x_sample_training = tf.random.uniform((N_x_training, 1), 0, L) # spatial interval (0, L)
t_sample_training = tf.random.uniform((N_t_training, 1), 0, T)  # time interval (0, T)
xt_sample_training = tf.concat([x_sample_training, t_sample_training], axis=1) # (0, L) x (0, T)
x_one_t_sample_training = tf.concat([tf.ones_like(t_sample_training), t_sample_training], axis=1) # x=1, 0<t<T
x_0_t_sample_training = tf.concat([tf.zeros_like(t_sample_training), t_sample_training], axis=1)  # x=0, 0<t<T

# test points
N_x_test = 100
N_t_test = 100

x_sample_test = tf.random.uniform((N_x_test, 1), 0, L)
t_sample_test = tf.random.uniform((N_t_test, 1), 0, T)
xt_sample_test = tf.concat([x_sample_test, t_sample_test], axis=1)
x_one_t_sample_test = tf.concat([tf.ones_like(t_sample_test), t_sample_test], axis=1)
x_0_t_sample_test= tf.concat([tf.zeros_like(t_sample_test), t_sample_test], axis=1)

### Losses for the ODE-PDE system

# initial conditions for $\psi$ and $h$

def psi_0(x): 
    return tf.sin(np.pi * x)

def loss_psi_0(sample_points):
    psi_init_pred = psi_net(tf.concat([sample_points, tf.zeros_like(sample_points)], axis=1))
    return tf.reduce_mean(tf.square(psi_init_pred - psi_0(sample_points)))
#print(loss_psi_0(x_sample_training))

h_0 = 0.5

def loss_h_0(h_0):
    t_init = tf.zeros((100, 1))
    h_init_pred = h_net(t_init)    
    return tf.sqrt(tf.reduce_mean(tf.square(h_init_pred - h_0)) + delta)
#loss_h_0(h_0)

# final conditions

def loss_psi_T(sample_points):
  psi_T_predict = psi_net(tf.concat([sample_points, T * tf.ones_like(sample_points)], axis=1))
  return tf.reduce_mean(tf.square(psi_T_predict)) # \psi(x, T)=0
#print(loss_psi_T(x_sample_training))


def loss_h_T(t_final): 
  t_final = t_final * np.ones((100, 1))
  h_final = np.ones((100, 1))
  h_final_pred = h_net(t_final)
  return tf.sqrt(tf.reduce_mean(tf.square(h_final_pred - h_final)) + delta) #k(T)=0 
#loss_h_T(T)

def loss_boundary_0(sample_points):
    x_zero_t_sample = tf.concat([tf.zeros_like(sample_points), sample_points], axis=1)
    # Computation of \psi_x(0,t)
    with tf.GradientTape(persistent=True) as tape3:
        tape3.watch(x_zero_t_sample)
        u_pred = u_net(x_zero_t_sample)
        psi_pred_boundary0 = psi_net(x_zero_t_sample)
        psi_x_boundary0 = tape3.gradient(psi_pred_boundary0, x_zero_t_sample)[:, 0:1]
    return tf.reduce_mean(tf.square(u_pred-psi_x_boundary0))
#print(loss_boundary_0(t_sample_training))

def loss_boundary_1(sample_points):
    x_one_t_sample = tf.concat([tf.ones_like(sample_points), sample_points], axis=1)  # points (1, t)
    psi_boundary1_pred = psi_net(x_one_t_sample)
    return tf.reduce_mean(tf.square(psi_boundary1_pred))

#print(loss_boundary_1(t_sample_training))

# constraint z>=0

def smooth_max(x1, x2, beta=10):
    return (1.0 / beta) * tf.math.log(tf.exp(beta * x1) + tf.exp(beta * x2))

x_1 = tf.constant([0.])
def loss_z_non_negative(sample_points): #sample points
    return tf.reduce_mean(smooth_max(x_1, - psi_net(sample_points))) 
#loss_z_non_negative(xt_sample_training)

# PDE: h(t) * psi_t - psi_xx - (x / 2) * h'(t) * psi_x - h(t) * psi * (1 - psi) = 0
   

def loss_pde(spatial_time_points, time_points):
        x_sample = spatial_time_points[:, 0]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(spatial_time_points)
            psi_pred = psi_net(spatial_time_points)
            psi_x = tape.gradient(psi_pred, spatial_time_points)[:, 0:1]
            psi_t = tape.gradient(psi_pred, spatial_time_points)[:, 1:2]
            psi_xx = tape.gradient(psi_x, spatial_time_points)[:, 0:1]
        with tf.GradientTape(persistent=True) as tape5:
            tape5.watch(time_points)
            h_pred = h_net(time_points)
            h_t_predict = tape5.gradient(h_pred, time_points)[:, 0:1]
        
        return tf.reduce_mean(tf.square(h_pred * psi_t - psi_xx - (x_sample / 2) * h_t_predict * psi_x - h_pred * psi_pred * (1 - psi_pred)))
        
#loss_pde(xt_sample_training, t_sample_training)  

# ODE: h^{\prime}(t)+2\mu \psi_{x}(1,t)=0
def loss_ode(time_points):
    one_t_points = tf.concat([tf.ones_like(time_points), time_points], axis=1)
    with tf.GradientTape() as tape2:
      tape2.watch(one_t_points)      
      psi_pred_boundary = psi_net(one_t_points)
      psi_x_boundary = tape2.gradient(psi_pred_boundary, one_t_points)[:, 0:1]
    with tf.GradientTape() as tape3:
       tape3.watch(time_points)
       h_predict = h_net(time_points)
       h_t = tape3.gradient(h_predict, time_points)[:, 0:1]

    return tf.reduce_mean(tf.square(h_t + 2 * mu * psi_x_boundary))
    
#loss_ode(t_sample_training)

# total loss function
def loss_add(sample_points, h_0, x_sample_points, t_sample_points, t_final):
    loss_init_psi = loss_psi_0(x_sample_points)
    loss_init_h = loss_h_0(h_0)
    loss_final_psi = loss_psi_T(x_sample_points)
    loss_final_h = loss_h_T(t_final)    
    loss_pde_psi = loss_pde(sample_points, t_sample_points)
    loss_dyn_h = loss_ode(t_sample_points)
    loss_boundary0 = loss_boundary_0(t_sample_points)
    loss_boundary1 = loss_boundary_1(t_sample_points)
    loss_z_constraint = loss_z_non_negative(sample_points)
    loss_total = 100 * loss_init_psi + 10 * loss_init_h + 100 * loss_final_psi + 15 * loss_final_h  + 0.001 * loss_boundary0 + 25 * loss_boundary1 + 15 * loss_pde_psi + 15 * loss_dyn_h  +  loss_z_constraint
    
    return loss_total, loss_init_psi, loss_init_h, loss_final_psi, loss_final_h, loss_pde_psi, loss_dyn_h, loss_boundary0, loss_boundary1, loss_z_constraint
    
#loss_add(xt_sample_training, h_0, x_sample_training, t_sample_training, T)

### Training

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
@tf.function
def train_step():
    with tf.GradientTape() as tape4:
        loss_total, loss_init_psi, loss_init_h, loss_final_psi, loss_final_h, loss_pde_psi, loss_dyn_h, loss_boundary0, loss_boundary1, loss_z_constraint =  loss_add(
            xt_sample_training, h_0, x_sample_training, t_sample_training, T
            )                          
    gradients = tape4.gradient(loss_total, psi_net.trainable_variables + h_net.trainable_variables + u_net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, psi_net.trainable_variables + h_net.trainable_variables + u_net.trainable_variables))
    loss_test, _, _, _, _, _, _, _, _, _ = loss_add(
            xt_sample_test, h_0, x_sample_test, t_sample_test, T
            ) 
    return loss_test, loss_total, loss_init_psi, loss_init_h, loss_final_psi, loss_final_h, loss_pde_psi, loss_dyn_h, loss_boundary0, loss_boundary1, loss_z_constraint
train_step()
 
epochs = 20000
iterations, loss_history_train, loss_history_test = list(), list(), list()

for epoch in range(epochs):
    loss_test, loss_total, loss_init_z, loss_init_k, loss_final_z, loss_final_k, loss_pde, loss_dyn_k, loss_boundary_0, loss_boundary_1, loss_z_constraint = train_step()
        
    if epoch % 1000 == 0:
        iterations.append(epoch)
        loss_history_train.append(loss_total)
        loss_history_test.append(loss_test)
        
        print(
            f"Epoch {epoch}: Loss_test = {loss_test.numpy():.4f}, Loss_total = {loss_total.numpy():.4f}, loss_init_z = {loss_init_z.numpy():.4f}, loss_init_k = {loss_init_k.numpy():.4f}, loss_final_z = {loss_final_z.numpy():.4f}, loss_final_k = {loss_final_k.numpy():.4f}, loss_pde = {loss_pde.numpy():.4f}, loss_dyn_k = {loss_dyn_k.numpy():.4f}, loss_boundary0 = {loss_boundary_0.numpy():.4f}, loss_boundary1 = {loss_boundary_1.numpy():.4f}, loss_z_constraint = {loss_z_constraint.numpy():.4f}"
            )
print(f"generalization error = {np.abs(loss_history_train[-1] - loss_history_test[-1])}")

### Post-processing

# using LaTeX in the pictures that follow
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX for all text
    "font.family": "serif",  # Use serif font (default LaTeX style)
    "font.size": 12  # Adjust font size
})

# convergence history

fig, ax = plt.subplots()
ax.plot(iterations, loss_history_train, '-', color='blue', linewidth=2, label=r'Train loss')
ax.plot(iterations, loss_history_test, '-', color='red', linewidth=2, label=r'Test loss')
ax.set(xlabel=r'epochs')
plt.legend(framealpha=1, shadow=True)
#plt.savefig('../figures/nonlinear_convergence_history.pdf', format='pdf')
plt.show()

# Picture of $\psi(x,t)$

# Network evaluation
Nx, Nt = 200, 200  # Number of points for x ant t
x = np.linspace(0, L, Nx).reshape(-1, 1)
t = np.linspace(0, T, Nt).reshape(-1, 1)
X, T_grid = np.meshgrid(x, t)


xt_eval = np.hstack((X.flatten()[:, None], T_grid.flatten()[:, None]))
z_eval = psi_net.predict(xt_eval).reshape(Nt, Nx)

#tf.reduce_all(z_eval >= 0.01)

# Visualization

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
surf=ax.plot_surface(X, T_grid, z_eval, cmap='jet', linewidth=0, antialiased=False)
ax.grid(False)
ax.set_zlim(0, 1)
ax.set_xlabel(r'$x$', fontsize=18, color="black", labelpad=20)
ax.set_ylabel(r'$t$', fontsize=18, color="black", labelpad=20)
#plt.savefig('../figures/state_psi_nonlinear_controlled.pdf', format='pdf')
plt.show()

Nt_discrete = 200 #100
Nx_discrete = 200 #100
t = np.linspace(0, T, Nt_discrete).reshape(-1, 1)

k_eval = h_net(t).numpy()
L_eval = np.sqrt(k_eval)

discrete_time = np.array([t[0], t[66], t[133], t[199]]) 
y = np.array([L_eval[1], L_eval[66], L_eval[133], L_eval[199]])
xx_plot = np.zeros((4, Nx_discrete))
z_new = np.zeros((4, Nx_discrete))
z_eval_discrete = np.zeros(Nx_discrete)
for i in range(4):       
    xx = np.linspace(0, y[i], Nx_discrete)    
    xx_plot[i, :] = xx.T   
    xx = xx / y[i] 
    tt =  discrete_time[i] * np.ones(len(xx))
    xt_eval_discrete = np.hstack((xx.flatten()[:, None], tt.flatten()[:, None]))
    z_eval_discrete = psi_net.predict(xt_eval_discrete).T   
    z_new[i, :] = z_eval_discrete
    
plt.figure()
plt.plot(xx_plot[0, :], z_new[0, :], label=r'$\varrho(:,$' +f'{np.round(discrete_time[0], 3)}' + r'$)$', linewidth=2, color='b')
plt.plot(xx_plot[1, :], z_new[1, :], label=r'$\varrho(:,$' +f'{np.round(discrete_time[1], 3)}' + r'$)$', linewidth=2, color='r')
plt.plot(xx_plot[2, :], z_new[2, :], label=r'$\varrho(:,$' +f'{np.round(discrete_time[2], 3)}' + r'$)$',linewidth=2, color='g')
plt.plot(xx_plot[3, :], z_new[3, :], label=r'$\varrho(:,$' +f'{np.round(discrete_time[3], 3)}' + r'$)$', linewidth=2, color='m')
plt.xlabel('$L(t)$')
plt.legend(framealpha=1, shadow=True)
#plt.savefig('../figures/nonlinear_densities.pdf', format='pdf')


# Visualization of L(t)
plt.figure()
plt.plot(t, L_eval, label=r'$L(t)$', linewidth=2, color='blue')
#plt.ylim(0, 0.8)
plt.xlabel(r'$t$', color="black", labelpad=20)
plt.legend(framealpha=1, shadow=True)
#plt.savefig('../figures/nonlinear_state_L.pdf', format='pdf')
plt.show()


# Computation and visualization of $\psi_x(0,t)$
x_0_t_sample = tf.concat([tf.zeros_like(t), t], axis=1)
with tf.GradientTape() as tape2:
    tape2.watch(x_0_t_sample)
    z_pred_boundary = psi_net(x_0_t_sample)
    z_x_boundary = tape2.gradient(z_pred_boundary, x_0_t_sample)[:, 0:1]
tt = np.linspace(0, T, 200).reshape(-1, 1)
plt.figure()
plt.plot(tt, z_x_boundary, label='control $u(t)$', linewidth=2, color='blue')
plt.xlabel('$t$', color="black", labelpad=20)
plt.legend(framealpha=1, shadow=True)
#plt.savefig('../figures/nonlinear_control_u.pdf', format='pdf')
plt.show()


