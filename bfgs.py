import numpy as np 
import matplotlib.pyplot as plt 

def f(x):
    '''
    FUNCTION TO BE OPTIMISED
    '''
    d = len(x)
    return sum(100*(x[i+1]-x[i]**2)**2 + (x[i]-1)**2 for i in range(d-1))

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
        a *= 0.5
        x_new = x + a * p 
        nabla_new = grad(f,x_new)
    return a


def BFGS(f,x0,max_it):
    '''
    DESCRIPTION
    BFGS Quasi-Newton Method, implemented as described in Nocedal:
    Numerical Optimisation.
    INPUTS:
    f:      function to be optimised 
    x0:     intial guess
    max_it: maximum iterations 
    plot:   if the problem is 2 dimensional, returns 
            a trajectory plot of the optimisation scheme.
    OUTPUTS: 
    x:      the optimal solution of the function f 
    '''
    d = len(x0) # dimension of problem 
    nabla = grad(f,x0) # initial gradient 
    H = np.eye(d) # initial hessian
    x = x0[:]
    it = 2 
    x_store =  np.array([np.zeros(d)])

    while np.linalg.norm(nabla) > 1e-5: # while gradient is positive
        if it > max_it: 
            print('Maximum iterations reached!')
            break
        it += 1
        p = -H@nabla # search direction (Newton Method)
        a = line_search(f,x,p,nabla) # line search 
        s = a * p 
        x_new = x + a * p 
        nabla_new = grad(f,x_new)
        y = nabla_new - nabla 
        y = np.array([y])
        s = np.array([s])
        y = np.reshape(y,(d,1))
        s = np.reshape(s,(d,1))
        r = 1/(y.T@s)
        li = (np.eye(d)-(r*((s@(y.T))))) #updating matrix of H_k left side
        ri = (np.eye(d)-(r*((y@(s.T))))) #updating matrix of H_k right side
        hess_inter = li@H@ri
        H = hess_inter + (r*((s@(s.T)))) # BFGS Update
        nabla = nabla_new[:] 
        x = x_new[:]
        x_store = np.append(x_store, [x_new], axis = 0)

    return x, x_store


x_opt, xstore= BFGS(f,[-1.2,1],100)
print(x_opt)
plt.scatter(xstore[:, 0], xstore[:, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.show()