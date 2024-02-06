# Testing parameter estimation

import numpy as np 
import matplotlib.pyplot as plt
 
from suppressedBD import *

###################################################

# Estimation functions

def EU(k,Y,theta): # expected number of births from state k in time t
    
    # Here, we regard k as the physical index (k = 0 absorbing)
    # a,b, are also regarded as physical indices (a,b ~ j)
    # the function P(i,j,t,b,g) takes unphysical indices ( i = -1 absorbing)
    
    if k == 0: # zero probability of another birth from zero state
        return (0,0)
    
    beta_ph,delta_ph,gamma_ph = theta 
    a,b,t_ph = Y
    
    # rescale from physical parameters to rescaled params in transition matrix expression
    beta = beta_ph/delta_ph
    gamma = gamma_ph/delta_ph
    t = delta_ph*t_ph
    
    # birth rate in physical state k
    lambdak = k*beta
    
    # rescale indices to 'unphysical'
    a = a-1
    b = b-1
    k = k-1
    
    pab = P(a,b,t,beta,gamma)
    
    integrand = lambda tau: P(a,k,tau,beta,gamma)*P(k+1,b,t-tau,beta,gamma)
    result,error = quadrature(integrand,0.00001,t)
    result = lambdak/pab *result
    
    rescaled_error = lambdak/pab*error
    
    return result,rescaled_error
    
def ED(k,Y,theta): # expected number of deaths from state k in time t

    # Here, we regard k as the physical index (k = 0 absorbing)
    # a,b, are also regarded as physical indices
    # the function P(i,j,t,b,g) takes unphysical indices ( i = -1 absorbing)
    
    beta_ph,delta_ph,gamma_ph = theta
    a,b,t_ph = Y
    
    if k == 0 or k == 1: # zero probability of another death from zero state, zero prob of absorption is b nonzero
        return (0,0)
    
    
    # rescale from physical parameters to rescaled params in transition matrix expression
    beta = beta_ph/delta_ph
    gamma = gamma_ph/delta_ph
    t = delta_ph*t_ph
    
    # birth rate in physical state k
    muk = k + gamma
    
    # rescale indices to 'unphysical' for when they are plugged in to transition matrix
    a = a-1
    b = b-1
    k = k-1
    
    pab = P(a,b,t,beta,gamma)
    
    integrand = lambda tau: P(a,k,tau,beta,gamma)*P(k-1,b,t-tau,beta,gamma)
    result,error = quadrature(integrand,0.00001,t)
    result = muk/pab *result
    
    rescaled_error = muk/pab*error
    
    return result,rescaled_error
    
def ET(k,Y,theta): # expected time spent in state k (expected absence of births/deaths)
    
    beta_ph,delta_ph,gamma_ph = theta
    a,b,t_ph = Y
    
    if k == 0: # zero probability of another death from zero state
        return (0,0)
        # What we return with k = 0 here is not important, it only enters as the product k*E(k,Y,theta)
    
    # rescale from physical parameters to rescaled params in transition matrix expression
    beta = beta_ph/delta_ph
    gamma = gamma_ph/delta_ph
    t = delta_ph*t_ph
    
    # rescale indices to 'unphysical' for when they are plugged in to transition matrix
    a = a-1
    b = b-1
    k = k-1
    
    pab = P(a,b,t,beta,gamma)

    print('pab',pab)
    
    integrand = lambda tau: P(a,k,tau,beta,gamma)*P(k,b,t-tau,beta,gamma)
    result,error = quadrature(integrand,0.00001,t)
    result = result/(pab*delta_ph)
    
    rescaled_error = error/(pab*delta_ph)
    
    return result,rescaled_error

################################################    
state_cutoff = 500 
################################################   

def ET_particle(Y,theta,cutoff = state_cutoff): # expected time spent by particle
    
    array = np.asarray([k*ET(k,Y,theta)[0] for k in range(cutoff)])
    
    return np.sum(array)

# parameter updates
pcutoff = 20 # no larger than this

def beta_update(Y,theta,cutoff = state_cutoff,tol = 1e-15): # update to birth rate parameter
    eu = 0
    et = ET_particle(Y,theta,cutoff = cutoff)
    for k in range(cutoff):
        term = EU(k,Y,theta)[0]
        if abs(term) < tol and k > max(Y[:2]): # cut off sum if terms very small
            return min(eu/et,pcutoff)
        else:
            eu += term
    return min(eu/et,pcutoff)

def delta_update(Y,theta,cutoff = state_cutoff,tol = 1e-15):
    beta,delta,gamma = theta
    def q(k):
        return k*delta/(delta*k + gamma)
    ed = 0
    et = ET_particle(Y,theta,cutoff = cutoff)
    for k in range(cutoff):
        term = q(k)*ED(k,Y,theta)[0]
        
        if abs(term) < tol and k > max(Y[:2]):
            return min(ed/et,pcutoff)
        else:
            ed += term
    return min(ed/et,pcutoff)


def gamma_update(Y,theta,cutoff = state_cutoff,tol = 1e-15):
    beta,delta,gamma = theta
    def q(k):
        return k*delta/(delta*k + gamma)
    t_ph = Y[2]
    beta,delta,gamma = theta
    eg = 0
    for k in range(cutoff):
        term = (1-q(k))*ED(k,Y,theta)[0]
        
        if abs(term) < tol and k > max(Y[:2]):
            return min(eg/t_ph,pcutoff)
        else:
            eg += term
    return min(eg/t_ph,pcutoff)
    

def thetaUpdate(obs,theta):
    bnew = beta_update(obs,theta)
    dnew = delta_update(obs,theta)
    gnew = gamma_update(obs,theta)
    return (bnew,dnew,gnew)
    

def BDestimate(observations): # main function
    theta = (1,1,1) # initialize parameters
    for obs in observations:
        print('observation',obs)
        print("params =",theta)
        theta = thetaUpdate(obs,theta)
    return theta


if __name__ == "__main__":

	observation = (37.0, 66.0, 0.7818563654384469)
	params = (0.9064116291835063, 0.5001657950855488, 0.28059351861724574)

	new = bd.thetaUpdate(observation,params)

	print(new)




	# # Generate synthetic test data
	# btest = 1.1
	# dtest = 1
	# gtest = 1.5
	# init = 20

	# sim = bd.BDensemble(btest,dtest,gtest,init,1000,1)

	# traj = sim.run_trajectory()
	# print("ran trajectory.")
	# js,Fs,ts = traj

	# # discretely observe the synthetic process
	# Nobs = 30

	# obsInds = []
	# for t in np.linspace(0,max(ts),Nobs+2):
	#     obsInds.append(np.argmin(np.abs(ts-t)))

	# observations = []
	# for i in range(Nobs):
	#     i1 = obsInds[i]
	#     i2 = obsInds[i+1]
	#     obs = (js[i1],js[i2],ts[i2] - ts[i1])
	#     if obs[2] > 0:
	#         observations.append(obs)
	#         print(obs)
	#     else: 
	#         continue
	    
	# # print(observations)
	# plt.plot(ts,js)
	# plt.scatter(ts[obsInds[1:-1]],js[obsInds[1:-1]],color = 'red',label = 'observations')
	# plt.legend()
	# plt.xlabel(r"time $t$")
	# plt.ylabel(r"population $j(t)$")
	# plt.title(r"$\beta = {},\ \delta = {},\ \gamma = {}$".format(btest,dtest,gtest))
	# plt.show()

	# cont = input("Continue? (Y/N)")
	# if cont == "N":
	# 	print("DONE.")
	# else:


	# 	bf,df,gf = bd.BDestimate(observations)

	# 	print("FINAL ESTIMATE:")
	# 	print("beta =",bf/df)
	# 	print("gamma =",gf/df)
	# 	print("Actual: beta = {}, gamma = {}".format(btest/dtest,gtest/dtest))












