from numpy import kaiser
import jax
import jax.numpy as np

def f(x,t):
    output = 0
    for i in range(t):
        output += x*i

    return output

def leapfrog_step(x, v, grad_potential, dt):
    v_half = v + 0.5*grad_potential(x)*dt
    x_new = x + v_half*dt
    v_new = v_half + 0.5*grad_potential(x_new)*dt
    return x_new, v_new

def potential(x1, m1, m2, x2):
    return m1*m2/np.sum((x1-x2)**2)

grad_potential = jax.grad(potential)

@jax.jit
def simulator_step(m1, m2, x1, x2, v1, v2, dt=0.1):
    v_half_1 = v1 + 0.5*grad_potential(x1, m1, m2, x2)*dt/m1
    x_new_1 = x1 + v_half_1*dt
    v_new_1 = v_half_1 + 0.5*grad_potential(x_new_1, m1, m2, x2)*dt/m1
    v_half_2 = v2 + 0.5*grad_potential(x2, m2, m1, x1)*dt/m2
    x_new_2 = x2 + v_half_2*dt
    v_new_2 = v_half_2 + 0.5*grad_potential(x_new_2, m2, m1, x1)*dt/m2
    return x_new_1, x_new_2, v_new_1, v_new_2

def simulator(m1, m2, x1, x2, v1, v2, step, max_t):
    dt = max_t/step
    x2_array = []
    for i in range(step):
        x1, x2, v1, v2 = simulator_step(m1, m2, x1, x2, v1, v2, dt)
        x2_array.append(x2)
    return x2_array

m1 = 100.
m2 = 1.
x1 = np.array([0.,0.,0.])
x2 = np.array([10.,0.,0.])
v1 = np.array([0.,0.,0.])
v2 = np.array([0.,1.42,0.])
step = 500
max_t = 200

data = np.array(simulator(m1, m2, x1, x2, v1, v2, step, max_t))

def loss(m1, m2, x1, x2, v1, v2, step, max_t):
    x2_array = np.array(simulator(m1, m2, x1, x2, v1, v2, step, max_t))
    return np.sum((x2_array-data)**2)

dLdt = jax.grad(loss,7)(m1, m2, x1, x2, v1, v2, step, max_t+0.1)