import numpy as np
import random as rd
import copy
def SBX_crossover(a,b,D,mu,task_bound):
    mu = mu
    lbound = task_bound[0]
    ubound = task_bound[1]
    child1 = np.zeros(D)
    child2 = np.zeros(D)
    u = np.random.uniform(0,1,size=D)
    cf = np.zeros(D)
    for i in range(D):
        if u[i]<=0.5:
            cf[i] = (2*u[i])**(1/(mu+1))
        else:
            cf[i] = (2*(1-u[i]))**(-1/(mu+1))
        tmp = 0.5 * ((1 - cf[i]) * a[i] + (1 + cf[i]) * b[i])
        child1[i] = tmp
        tmp = 0.5 * ((1 + cf[i]) * a[i] + (1 - cf[i]) * b[i])
        child2[i] = tmp
    child1 = np.clip(child1,0,1)
    child2 = np.clip(child2, 0, 1)
    return child1, child2


def poly_mutation(pop,D,mu,task_bound):
    mu = mu
    lbound = task_bound[0]
    ubound = task_bound[1]
    offsp = copy.deepcopy(pop)
    for i in range(D):
        rand1 = np.random.rand()
        if rand1 < (1/D):
            rand2 = np.random.rand()
            if rand2 < 0.5:
                tmp1 = (2 * rand2) ** (1 / (1 + mu)) - 1
                tmp2 = pop[i] + tmp1*pop[i]
                offsp[i] = tmp2
            else:
                tmp1 = 1-(2*(1-rand2))**(1/(1+mu))
                tmp2 = pop[i] + tmp1*(1-pop[i])
                offsp[i] = tmp2
        else:
            offsp[i] = pop[i]
    offsp = np.clip(offsp,0,1)
    return offsp

def DE_rand_1(x, p1, p2, p3, F, cr, d,bound):
    lb = bound[0]
    ub = bound[1]
    v = p1 + F * ( p2 - p3 )
    u = DE_crossover( x, v, cr, d)
    u = np.clip(u, lb,ub)
    return u
def DE_crossover(x,v,cr,d):
    u = copy.deepcopy(v)
    k = np.random.randint(0, d)
    for i in range(d):
        if np.random.rand()<cr or i==k:
            u[i] = v[i]
        else:
            u[i] = x[i]
    return u

