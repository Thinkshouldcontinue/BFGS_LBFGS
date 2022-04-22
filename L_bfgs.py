import numpy as np 
import matplotlib.pyplot as plt 

def f(x):
    '''
    FUNCTION TO BE OPTIMISED
    '''
    d = len(x)
    return sum(100*(x[i+1]-x[i]**2)**2 + (x[i]-1)**2 for i in range(d-1))

#Calculate the gradient of this function:
def grad(f,x): 
    '''
    CENTRAL FINITE DIFFERENCE CALCULATION
    '''
    h = np.cbrt(np.finfo(float).eps)
    d = len(x)
    nabla = np.zeros(d)
    for i in range(d): 
        x_for = np.copy(x) 
        x_back = np.copy(x)
        x_for[i] += h 
        x_back[i] -= h 
        nabla[i] = (f(x_for) - f(x_back))/(2*h) 
    return nabla 

def line_search(f,x,p,nabla):
    '''
    BACKTRACK LINE SEARCH WITH WOLFE CONDITIONS
    '''
    a = 1
    c1 = 1e-4 
    c2 = 0.9 
    fx = f(x)
    x_new = x + a * p 
    nabla_new = grad(f,x_new)
    while f(x_new) >= fx + (c1*a*nabla.T@p) or nabla_new.T@p <= c2*nabla.T@p : 
        a = a * 0.9
        x_new = x + a * p 
        nabla_new = grad(f,x_new)
    return a

def recursion_two_loop(gradient, s_stored, y_stored, m):
    q = gradient
    length = len(q)
    a = np.zeros(m)
    rou = np.array([1/np.dot(y_stored[j, :], s_stored[j, :]) for j in range(m)])
    for i in range(m):
        a[m - 1 - i] = rou[m - 1 - i] * np.dot(s_stored[m - 1 - i, :], q)
        q = q - a[m - 1 - i]*y_stored[m - 1 - i, :]
    
    H_k0 = (np.dot(s_stored[m - 1], y_stored[m - 1])/np.dot(y_stored[m - 1], y_stored[m - 1]))
    r = H_k0 * q
    
    for i in range(m):
        beta = rou[i] * np.dot(y_stored[i, :], r)
        r = r + (a[i] - beta) * s_stored[i]
    return r

def L_bfgs(f, x0, max_it, m):
    '''
    INPUTS:
    f:      function to be optimised 
    x0:     intial guess
    max_it: maximum iterations 
    OUTPUTS: 
    x:      the optimal solution of the function f 
    '''
    d = len(x0) # dimension of problem 
    nabla = grad(f,x0) # initial gradient 
    x = x0[:]
    x_store =  np.array([x0])

    '''
    Store the {y_i, s_i}
    '''
    y_stored = []
    s_stored = []
    p = - nabla
    alpha = line_search(f,x,p,nabla)
    s_stored.append(alpha * p)
    grad_old = nabla[:]
    x = x + alpha * p
    nabla = grad(f, x)
    y_stored.append(nabla - grad_old)
    m_ = 1
    it = 1
    x_store = np.append(x_store, [x], axis = 0)
    while np.linalg.norm(nabla) > 1e-5: # while gradient is positive
        if it > max_it: 
            print('Maximum iterations reached!')
            break

        if 0 < it and it < m :
            p = - recursion_two_loop(nabla, np.array(s_stored), np.array(y_stored), m_)
            alpha = line_search(f,x,p,nabla)
            s_stored.append(alpha * p)
            grad_old = nabla[:]
            x = x + alpha * p
            nabla = grad(f, x)
            y_stored.append(nabla - grad_old)
            m_ = m_ + 1
            it = it + 1
            x_store = np.append(x_store, [x], axis = 0)
            
        else:
            p = - recursion_two_loop(nabla, np.array(s_stored), np.array(y_stored), m)
            alpha = line_search(f,x,p,nabla)

            #append the s_k+1 
            s_stored.append(alpha * p)

            #discard the s_(k-m)
            s_stored.pop(0)
            grad_old = nabla[:]
            x = x + alpha * p
            nabla = grad(f, x)

            #append the y_k+1
            y_stored.append(nabla - grad_old)

            #discard the y_k-m
            y_stored.pop(0)
            it = it + 1

            x_store = np.append(x_store, [x], axis = 0)
    
    return x, x_store

x_opt, xstore= L_bfgs(f,[-1.5,5], 100, 10)

'''
print(x_opt)
plt.scatter(xstore[:, 0], xstore[:, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.show()
'''

print('optimal value:', x_opt)
fig, ax = plt.subplots()
def animate(i):
    length = len(xstore[:, 0])
    ax.scatter(xstore[:, 0][i], xstore[:, 1][i])
    ax.yaxis.set_ticks(np.arange(-2.5, 3, 0.25))
    ax.xaxis.set_ticks(np.arange(-2.5, 3, 0.25))

ani = FuncAnimation(fig, animate, frames= len(xstore[:, 0]), interval=500, repeat=False)
plt.show()
