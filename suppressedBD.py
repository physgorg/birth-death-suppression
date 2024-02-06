# The suppressed birth and death process

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import ticker
import scipy.special as sc
from matplotlib import rcParams
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.integrate import quadrature

import re
import mpmath
from tqdm import tqdm
import multiprocessing

from numpy.lib.scimath import sqrt

# this function renders strings like "hello $n=5$ world" into LaTeX, for plots
# note that in Python, you must use double backslash "$\\beta$" to 'escape' the first one
# GPT-4 wrote this function, thanks GPT!
# The reason for doing this (instead of r'latex here') is to support different fonts in mpl renderings
def process_latex(text):
	# Split the input text into regular and LaTeX parts
	parts = re.split(r'(\$.*?\$)', text)

	# Replace each LaTeX part with its formatted version
	for i, part in enumerate(parts):
		if part.startswith('$') and part.endswith('$'):
			latex_expression = part[1:-1]  # Remove the $$ delimiters
			parts[i] = f"${latex_expression}$"

	# Join the parts back into a single string
	combined_text = ''.join(parts)
	return combined_text

##########################################


# Auxiliary functions

def G(x): # gamma function
	return sc.gamma(x)

def F(a,b,c,z):

	# return np.real(sc.hyp2f1(a,b,c,z))
	if a == b:
	    return np.real((1-z)**(-1*a)*sc.hyp2f1(a,c-b,c,z/(z-1)))
	else:
	    return np.real(sc.hyp2f1(a,b,c,z))

def incBeta(z,a,b):
	return sc.beta(a,b)*sc.betainc(a,b,z)

def sigma(t,b):
	return np.exp((b-1)*t)

def z(t,b): # N = 1 absorption probability
	if b != 1:
		return (1-sigma(t,b))/(1-b*sigma(t,b))
	else:
		return t/(1+t)

def bigX(t,b): # auxiliary function in transition matrix
	if b != 1:
		return (b*sigma(t,b)-1)*(sigma(t,b)-b)/(b*(sigma(t,b)-1)**2)
	elif b == 1:
		return 1 - 1/t**2

def auxiliaryS(i,j,t,b,g): # element of population transition matrix
	# take logarithms here to prevent overflows
	prf = sc.loggamma(i+j+g+2) - sc.loggamma(j+1) - sc.loggamma(i+g+2)
	inside = F(-1*i,-1*j,-1-i-j-g,bigX(t,b))
	hyper = np.real(np.log(inside))
	return np.exp(prf+hyper)


def pi(n,b,g,ret_log = False): # potential coefficients
	if n == 0: # base case
		return 1
	elif n < 100:
		return b**n*G(n+1)/sc.poch(g+2,n)
	elif n >= 1000: # use asymptotic formula; still numerically rough
		coeff = G(g+2)
		term1 = np.exp((n-1)*np.log(b) - (g+1)*np.log(n))
		term2 = -1/2*g*(g+1)*np.exp((n-1)*np.log(b) - (g+2)*np.log(n))
		term3 = 1/24*g*(g+1)*(g+2)*(1+3*g)*np.exp((n-1)*np.log(b) - (g+3)*np.log(n))
		term4 = -1/48*g**2*(1+g)**2*(g+2)*(g+3)*np.exp((n-1)*np.log(b) - (g+4)*np.log(n))
		return coeff*(term1 + term2 + term3 + term4)
	else:
		# print(n)
		toexp = n*np.log(b)*sc.loggamma(g+2) + sc.loggamma(n+1)-sc.loggamma(g+n+2)
		if ret_log:
			return toexp
		else:
			return np.exp(toexp)

def auxiliaryF(k,l,t,b,g):

	# if isinstance(k,np.ndarray):


	# elif isinstance(l,np.ndarray):

	# else:
	# 	if k != l:
	coeff = l*np.log(b) + sc.loggamma(k+l+g+2) - sc.loggamma(k+1) - sc.loggamma(l+g+2)
	prod = np.log(np.real(F(-1*k,-1*l,-1-k-l-g,bigX(t,b))))
	return coeff + prod
		# else:



def P(i,j,t,b,g):

	order1_part = np.log(1/sigma(t,b)*(z(t,b))**(i+j)*(1-z(t,b))**(g+2))

	other = auxiliaryF(i,j,t,b,g)

	if np.any(np.isnan(order1_part)):
		print('order1 is NAN')
		print("i,j,t,b,g")
		print(i,j,t,b,g)
	if np.any(np.isnan(other)):
		print('other is NAN')
		print("i,j,t,b,g")
		print(i,j,t,b,g)
	
	return np.exp(order1_part + other)

#########################################
# Lifetime statistics 
#########################################

# probability of absorption
def absorbProb(t,N,b,g):
	return sc.betainc(N,g+1,z(t,b))

# asymptotic probability of absorption
def asymAbsorb(N,b,g):
	if b <= 1:
		return 1
	else:
		return sc.betainc(N,g+1,1/b)

# distribution of absorption times
def lifetimeMeasure(t,N,b,g):

	if b < 1:

		return (1-z(t,b))**(g+2)*(z(t,b))**(N-1)/(sigma(t,b)*incBeta(1,N,g+1))

	elif b == 1:

		 return (1/(1+t))**(g+2)*(t/(1+t))**(N-1)/(incBeta(1,N,g+1))

	elif b > 1:

		return (1-z(t,b))**(g+2)*(z(t,b))**(N-1)/(sigma(t,b)*incBeta(1/b,N,g+1))

# same, but with z as an explicit variable. Just a beta distribution
def z_lifetimeMeasure(z,N,g):

	return (1-z)**(g)*z**(N-1)/sc.beta(N,g+1)

# median lifetime function
def medianLifetime(N, bs, g):
	def scalar_medianLifetime(N, b, g):
		if b <= 1:
			zm = sc.betaincinv(N, g + 1, 0.5)
		elif b > 1:
			zm = sc.betaincinv(N, g + 1, 0.5 * sc.betainc(N, g + 1, 1 / b))
		tm = 1 / (b - 1) * np.log((1 - zm) / (1 - b * zm))
		return tm

	vectorized_medianLifetime = np.vectorize(scalar_medianLifetime)
	res = vectorized_medianLifetime(N, bs, g)
	return res

# average lifetime, computed via quad integration
def averageLifetime(N,bs,g):
	# uses quad integration 

	res = []
	for b in bs:
		# print(b)
		def integrand(t):
			return t*lifetimeMeasure(t,N,b,g)

		# print(integrand(np.linspace(1,100,10)))

		result, error = quad(integrand,0,100)

		res.append(result)

	return res

#########################################
# Footprint-related auxiliary functions

def Disc(x,b): # discriminant
	if isinstance(x,np.ndarray):
		interior = np.asarray(x**2*(b+1)**2 - 4*b,dtype = complex)
		return np.sqrt(interior	)
	else:
		return np.sqrt(complex(x**2*(b+1)**2 - 4*b))

def Ib(b): # bound of orthogonality zone
	return 2*np.sqrt(b)/(1+b)

def rtU(x,b): # root of quadratic 
	return x*(b+1)/2 + Disc(x,b)/2

def rtV(x,b): # root of quadratic
	return x*(b+1)/2 - Disc(x,b)/2

def cfA(x,b,g): # partial fraction decomposition
	if g == 0:
		return -1
	else:
		return -1*(g+2)/2 - x*g*(b-1)/(2*Disc(x,b))

def cfB(x,b,g): # partial fraction decomposition
	if g == 0:
		return -1
	else:
		return -1*(g+2)/2 + x*g*(b-1)/(2*Disc(x,b))

def firewalkW(n,x,b,g): # firewalk polynomials
	# this function needs to use the mpmath 2F1 function, as it supports complex parameter values.
	n = int(n)
	if b == 1:
		coeff = sc.poch(g+2,n)/sc.factorial(n)
		hyper = mpmath.hyp2f1(-1*n,n+g+2,1/2*(g+3),(1-x)/2,zeroprec = 1e-40)
		return np.real(complex(coeff*hyper))
	else:
		coeff = sc.poch(g+2,n)/sc.factorial(n)*rtU(x,b)**(-1*n)
		hyper = mpmath.hyp2f1(-1*n,-1*cfB(x,b,g),g+2,-1*rtU(x,b)*Disc(x,b)/b,zeroprec = 1e-40)
		return np.real(complex(coeff*hyper))   

def contMeasure(x,b,g): # continuous measure of orthogonality for firewalk polynomials
	if b == 1:
		prefactor = G(g/2+2)/(np.sqrt(np.pi)*G((g+3)/2))
		xpart = (1-x**2)**((g+1)/2)
		return prefactor*xpart
	elif abs(x) > Ib(b):
		return 0
	else:
		prefactor = (b + g+ 1)/(2*np.pi*1j)
		uvpart = rtU(x,b)**(cfA(x,b,g))*rtV(x,b)**(cfB(x,b,g))*(rtU(x,b)-rtV(x,b))**(-1*cfA(x,b,g))/((rtV(x,b)-rtU(x,b))**(cfB(x,b,g)+1))
		betapart = G(-1*cfA(x,b,g))*G(-1*cfB(x,b,g))/G(g+2)
		return np.real(prefactor*uvpart*betapart)

def atomX(k,b,g): # location of discrete measure Ib <= x_k <= 1
	numerator = b*(g+2*(k+1))**2
	denominator = (g + (k+1)*(b+1))*(b*g + (k+1)*(b+1))
	return np.sqrt(numerator/denominator)

def weightD(k,b,g): # weight of discrete measure at x_k
	if g == 0 or b == 1:
		return 0
	elif b > 1:
		xk = atomX(k,b,g)
		prefactor = (b+g+1)/(b**(g+3))
		uvpart = (rtU(xk,b))**(-1*k)*(rtV(xk,b))**(k+g+2)
		later_part = sc.poch(g+2,k)/sc.factorial(k)*Disc(xk,b)**(g+4)/(2*g*(b-1))
		return prefactor*uvpart*later_part
	else:
		return 0
	
def const_h(n,b,g): # potential coefficients for firewalk polynomials
	if n == 0:
		return 1
	else:
		prefactor = sc.poch(g+2,n)/(b**n*sc.factorial(n))
		fraction = (1+b+g)/(g+(b+1)*(n+1))
		return prefactor*fraction

def S(n,i,j,b,g,kmax = 50): # transition matrix for embedded jump chain
	if n <= 0:
		return 0
	if n % 2 == 0 and not i% 2 == j % 2:
		return 0 # zero if n is even and i,j are off-parity
	elif n % 2 != 0 and i % 2 == j % 2:
		return 0 # zero if n is odd and i,j are on-parity
	else:
		prf_1 = 1/const_h(j,b,g)
		integrand = lambda x: x**n*firewalkW(i,x,b,g)*firewalkW(j,x,b,g)*contMeasure(x,b,g)
		ib = Ib(b)
		cont_part,cont_error = quad(integrand,-1*ib,ib)
		if b == 1 or g == 0:
			return np.real(prf_1*cont_part)
		else:
			vals = [atomX(k,b,g)**n*weightD(k,b,g)*(firewalkW(i,atomX(k,b,g),b,g)*firewalkW(j,atomX(k,b,g),b,g) +firewalkW(i,-1*atomX(k,b,g),b,g)*firewalkW(j,-1*atomX(k,b,g),b,g))  for k in range(kmax+1)]
			summed = np.sum(vals)
			return np.real(prf_1*cont_part) + np.real(prf_1*summed)
	
def RnProb(n,bigN,b,g): # jump chain 'absorption' probabilities
	if n < bigN:
		return 0
	else:
		print('here')
		return (g+1)/(b+g+1)*S(n-1,bigN,0,b,g)

if __name__ == '__main__':
	print(RnProb(16,5,1.1,1))

def ftptIntegrate(func,b,g,kmax = 50,split_range = False): # integrate arbitrary kernel against measure
	# note: a custom integrator should be used here, ideally

	integrand = lambda x: func(x)*contMeasure(x,b,g)
	ib = Ib(b)
	if not split_range:
		cont_part,cont_error = quad(integrand,-1*ib,ib)
	elif split_range:
		im = 0.8*ib
		c1,e1 = quad(integrand,-1*ib,-1*im)
		c2,e2 = quad(integrand,-1*im,im)
		c3,e3 = quad(integrand,im,ib)
		cont_part = c1 + c2 + c3
	if b <= 1:
		return np.real(cont_part)
	else:
		vals1 = [func(atomX(k,b,g))*weightD(k,b,g) for k in range(kmax+1)]
		summed1 = np.sum(vals1)
		
		vals2 = [func(-1*atomX(k,b,g))*weightD(k,b,g) for k in range(kmax+1)]
		summed2 = np.sum(vals2)
		return np.real(cont_part) + np.real(summed1 + summed2)

def BurnProb(bigN,J,b,g): # escape probability of footprint reaching a given size

	if J < bigN:

		return 1

	else:
	
		kernel = lambda x: (g+1)/(b+g+1)*(1-x**(2*J-bigN))/(1-x)*firewalkW(bigN-1,x,b,g)
		
		return 1-ftptIntegrate(kernel,b,g,split_range = True)


#########################################

# CLASSES

# custom plot class. not really that much accomplished by using this :(
class MP:
	def __init__(self):
		
		rcParams.update({
						 "font.size": 14,
						# "text.usetex": True,
						"font.family":"Prima Sans"})

		self.fig, self.ax = plt.subplots()

		# Remove grid
		self.ax.grid(False)

		# Initialize an empty legend
		self.legend_entries = []

	def add_plot(self, x, y, filled = False,label=None, **kwargs):
		plot_obj = self.ax.plot(x, y, label=label, **kwargs)

		if filled != False:
			self.ax.fill_between(x,y,alpha = 0.5,color =filled)

		if label is not None:
			self.legend_entries.append(label)

		return plot_obj


	def set_title(self, title):
		self.ax.set_title(title)

	def set_xlabel(self, xlabel):
		self.ax.set_xlabel(xlabel)

	def set_ylabel(self, ylabel):
		self.ax.set_ylabel(ylabel)

	def set_xlim(self, xmin, xmax):
		self.ax.set_xlim(xmin, xmax)

	def set_ylim(self, ymin, ymax):
		self.ax.set_ylim(ymin, ymax)

	def show_legend(self,loc = 'lower right',**kwargs):
		self.ax.legend(self.legend_entries,loc = loc,**kwargs)
		self.ax.legend().get_frame().set_linewidth(2)

	def set_xticks(self, ticks, labels, **kwargs):
		self.ax.set_xticks(ticks,labels,**kwargs)
		# if labels is not None:
		#     self.ax.set_xticklabels(labels, )

	def set_yticks(self, ticks, labels=None, **kwargs):
		self.ax.set_yticks(ticks,labels = labels)
		# if labels is not None:
		#     self.ax.set_yticklabels(labels, **kwargs)


	def save_fig(self, file_name=None):
		if file_name is not None:
			self.fig.savefig(file_name, format="pdf",bbox_inches = 'tight')


	def add_letter_to_corner(self, letter, direction,offset = 0.1, xyvals = (0,0),fontsize=30):
		latex_letter = str(letter)
		# offset = 0.05
		if direction.lower() == 'north':
			xy = (0.5, 1 - offset)
			align = ('center', 'top')
		elif direction.lower() == 'south':
			xy = (0.5, offset)
			align = ('center', 'bottom')
		elif direction.lower() == 'east':
			xy = (1 - offset, 0.5)
			align = ('right', 'center')
		elif direction.lower() == 'west':
			xy = (offset, 0.5)
			align = ('left', 'center')
		elif direction.lower() == 'northeast':
			xy = (1 - offset, 1 - offset)
			align = ('right', 'top')
		elif direction.lower() == 'northwest':
			xy = (offset, 1 - offset)
			align = ('left', 'top')
		elif direction.lower() == 'southeast':
			xy = (1 - offset, offset)
			align = ('right', 'bottom')
		elif direction.lower() == 'southwest':
			xy = (offset, offset)
			align = ('left', 'bottom')
		else:
			raise ValueError('Invalid direction. Choose from north, south, east, west, northeast, northwest, southeast, southwest.')
		
		self.ax.annotate(latex_letter, xy=xy, xycoords='axes fraction',
						 fontsize=fontsize, ha=align[0], va=align[1],
						 xytext=xyvals, textcoords='offset points')


		def show(self):
			plt.show()

# simulating birth-death processes in ensemble
class BDensemble:
	
	def __init__(self,beta,delta,gamma,initial,sizeJ,ensembleN,F0 = None,t0 = 0):
		self.b = beta
		self.d = delta
		self.g = gamma
		self.N = initial
		self.tot = ensembleN
		self.J = sizeJ

		if F0 == None:
			self.F0 = initial 
			self.initial_state = np.vstack((self.N*np.ones((2,self.tot)),t0*np.ones(self.tot))).T
		else:
			self.F0 = F0
			jfpart = np.vstack((self.N*np.ones(self.tot),self.F0*np.ones(self.tot)))
			self.initial_state = np.vstack((jfpart,t0*np.ones(self.tot))).T
		
		
		
		
	def update(self,statearr): # one-transition update
		
		jvs,Fvs,tvs = statearr.T
		
		Nval = len(jvs)
		ones = np.ones(Nval)
		# aggregate birth/death rates
		lambdas = self.b*jvs
		mus = self.d*jvs + self.g*ones

		# propagate time
		tscales = 1/(lambdas + mus)
		delta_ts = np.random.exponential(tscales)


		# update population & footprint
		ps = lambdas*tscales
		delta_fs = np.random.binomial(ones.astype(np.int64),ps)
		delta_js = 2*delta_fs - ones

		delta = np.vstack((delta_js.astype(np.int64),delta_fs.astype(np.int64),delta_ts))
		
		return statearr + delta.T

	def run_trajectory(self,v = False): # run a single process up and out
		state = np.asarray(self.initial_state[:1])
		history = state
		alive = True # true if j > 0
		bounded = True # true if F < J
		while alive and bounded:
			state = self.update(state)
			j,F,t = state[0]
			if j == 0:
				alive = False
				if v: print('burned out (absorbed).')
			if F >= self.J:
				bounded = False
				if v: print('out of control (saturated).')
				
			history = np.vstack((history,state))           
		return history.T

	def run_finite_trajectory(self,T,v = False):
		state = np.asarray(self.initial_state[:1])
		history = state
		alive = True # true if j > 0
		bounded = True # true if F < J
		time_left = True
		while alive and bounded and time_left:
			state = self.update(state)
			j,F,t = state[0]
			if t > T: 
				time_left = False

			if j == 0:
				alive = False
				if v: print('burned out (absorbed).')
			if F >= self.J:
				bounded = False
				if v: print('out of control (saturated).')
			if time_left:	
				history = np.vstack((history,state))           
		return history.T
			
	def process_state(self,arr): # determine how many processes burned out, burned out of control a
		mask = np.logical_and(arr[:, 0] != 0, arr[:, 1] < self.J)
		out = arr[mask]
		died_out = np.sum(arr[:, 0] == 0)
		out_of_control = np.sum(arr[:, 1] >= self.J)
		died_out_times = arr[arr[:, 0] == 0, 2]
		out_of_control_times = arr[arr[:, 1] >= self.J, 2]
		return out, [died_out, out_of_control], [died_out_times, out_of_control_times]

	def ftpt_process(self,arr):
		mask = np.logical_and(arr[:, 0] != 0, arr[:, 1] < self.J)
		active = np.where(mask)
		# print(active)
		absorbed = np.where(arr[:, 0] == 0)
		escaped = np.where(arr[:, 1] >= self.J)
		return active,absorbed,escaped
	
	def run_ensemble(self,ret_times = False):
	
		initialized = self.initial_state
		active = initialized
		ndied = 0
		nburned = 0
		died_times = []
		out_of_control_times = []
		while len(active) > 0:
			active = self.update(active)
			active,nt,tt = self.process_state(active)
			ndied += nt[0]
			nburned += nt[1]
			if len(tt[0])>0:
				died_times += list(tt[0])
			if len(tt[1])>0:
				out_of_control_times += list(tt[1])
		if ret_times:
			return [ndied,nburned],[died_times,out_of_control_times]
		else:
			return [ndied,nburned]

	def run_FtptHist(self):
		initialized = self.initial_state
		time_hist = np.array([np.zeros(self.tot)])
		pop_hist = np.array([np.zeros(self.tot)])
		ftpt_hist = np.array([self.N*np.ones(self.tot)])
		active = initialized
		resarr = []
		counter = 0
		while len(active) > 0:
			active = self.update(active)
			active_inds,absorbed_inds,escaped_inds = self.ftpt_process(active)
			# print('active',active_inds)
			# print('absorbed',absorbed_inds)
			# print('escaped',escaped_inds)
			active = active[active_inds]
			js,Fs,ts = active.T

			dts = np.zeros(self.tot)
			dts[active_inds] = ts
			dts[absorbed_inds] = np.mean(ts)
			dts[escaped_inds] = np.mean(ts)
			time_hist = np.vstack((time_hist,dts))

			dfs = np.zeros(self.tot)
			dfs[active_inds] = Fs
			dfs[absorbed_inds] = ftpt_hist[-1][absorbed_inds]
			dfs[escaped_inds] = self.J
			ftpt_hist = np.vstack((ftpt_hist,dfs))

			djs = np.zeros(self.tot)
			djs[active_inds] = js
			djs[absorbed_inds] = ftpt_hist[-1][absorbed_inds]
			djs[escaped_inds] = self.J
			ftpt_hist = np.vstack((ftpt_hist,djs))



		return time_hist,ftpt_hist






##################################################################

# Plotting functions used in paper


def absorbPlotZeroG():
	t = np.linspace(0,30,500)

	plot = MP()

	N = 5
	
	# plot.set_title(process_latex("$\\beta = 0.8$"))

	plot.set_xlabel(process_latex("Time ($t$)"))
	plot.set_ylabel("Absorption probability")

	plot.add_plot(t,absorbProb(t,N,0.8,0),color = 'forestgreen',label = process_latex("$\\beta = 0.8$"),)
	plot.add_plot(t,absorbProb(t,N,1,0),color = 'orange',label = process_latex("$\\beta = 1$"))
	plot.add_plot(t,absorbProb(t,N,1.2,0),color = 'firebrick',label = process_latex("$\\beta = 1.2$"))
	plot.add_plot(t,np.ones(t.shape),color = 'black',linestyle = 'dashed')
	plot.show_legend()
	# plot.ax.legend().get_frame().set_linewidth(2)
	plot.set_ylim(0,1.05)
	plot.set_xlim(0,50)
	# colorbar = plt.colorbar(contour_filled)
	# colorbar.set_label('Median Lifetime')
	# plot.ax.axvline(x=1, linestyle='--', color='black')

	# plot.add_letter_to_corner("B")

	# plot.save_fig("zerog_absorb_probs.pdf")
	plt.show()

def absorbPlot():

	tmax = 30
	t = np.linspace(0,tmax,500)

	plot = MP()

	N = 5
	
	# plot.set_title(process_latex("$\\beta = 0.8$"))

	plot.set_xlabel(process_latex("Time ($t$)"))
	plot.set_ylabel("Absorption probability")

	gvals = [0,1,2,5]
	gvals.reverse()
	bet = 1.2

	for g in gvals:
		plot.add_plot(t,absorbProb(t,N,bet,g),label = process_latex("$\\gamma = {}$".format(g)))
	plot.add_plot(t,np.ones(t.shape),color = 'black',linestyle = 'dashed')
	plot.show_legend()
	# plot.ax.legend().get_frame().set_linewidth(2)
	plot.set_ylim(0,1.05)
	plot.set_xlim(0,tmax)
	# colorbar = plt.colorbar(contour_filled)
	# colorbar.set_label('Median Lifetime')
	# plot.ax.axvline(x=1, linestyle='--', color='black')

	plot.add_letter_to_corner("B",'south',offset = 0.03)

	plot.save_fig("super_crit_gam_absorb.pdf")
	plt.show()


def betaPlot():
	plot = MP()

	z = np.linspace(0,1,200)

	params = [[10,1],[1,10],[10,10]]

	for i in range(len(params)):
		Nval, gval = params[i]

		vs = [z_lifetimeMeasure(zz,Nval,gval) for zz in z]

		plot.add_plot(z,vs,label = process_latex("$N = {},\\ \\gamma = {}$".format(Nval,gval)))
		plot.ax.fill_between(z,vs,alpha = 0.7)
	plot.show_legend()
	plot.set_xlim(0,1)
	plot.set_xlabel(process_latex("$z(T)$"))
	plot.set_xticks([0,0.2,0.8,1],['0','early times','late times','1'])
	plot.set_ylim(0,8)
	plot.set_ylabel('Probability density')

def LifeMeasurePlot():

	plot = MP()

	N = 50

	t = np.linspace(0,30,500)


	plot.set_xlabel(process_latex("Absorption time"))
	plot.set_ylabel("Probability density")

	plot.add_plot(t,lifetimeMeasure(t,N,0.5,0),color = 'blue',label = process_latex("$\\beta = 0.5$"))
	plot.add_plot(t,lifetimeMeasure(t,N,1,0),color = 'gray',label = process_latex("$\\beta = 1$"))
	plot.add_plot(t,lifetimeMeasure(t,N,1.5,0),color = 'red',label = process_latex("$\\beta = 1.5$"))

	
	plot.show_legend()

	plot.set_ylim(0,0.31)
	plot.add_letter_to_corner("B",'southeast')

	plot.save_fig('zerog_n{}_lifetimes.pdf'.format(N))

	plt.show()


def FirewalkMeasurePlot():

	betas = [0.5,1,2]

	gvals = [0,2,4]

	fig, axs = plt.subplots(nrows=1, ncols=len(betas),figsize = (8,3))


	for i,b in enumerate(betas):
		ib = Ib(b)
		x = np.linspace(-1*ib,ib,500)
		axs[i].set_title(process_latex("$\\beta = {}$".format(b)))
		axs[i].set_ylim([0,1.5])
		axs[i].set_xlim([-1,1])
		axs[i].axhline(0,color = 'k', linewidth = 0.5)

		axs[i].set_xticks([-1,0,1],[-1,0,1])
		axs[i].vlines([-1*ib,ib],-1,2,color = 'k',linestyle = 'dashed',linewidth = 0.5)
		axs[i].set_yticks([])

		for g in gvals:
			ys = [contMeasure(xx,b,g) for xx in x]
			axs[i].plot(x, ys,label = process_latex("$\\gamma = {}$".format(g)))
			# axs[i].fill_between(x,ys,alpha = 0.6)
			
	handles, labels = axs[0].get_legend_handles_labels()
	# Create a single legend outside the subplots
	leg =  fig.legend(handles, labels,loc = 'right')
	leg.get_frame().set_linewidth(2)  
	fig.tight_layout(rect=[0, 0,0.9, 1])     

	# plt.savefig("measures.pdf",bbox_inches = 'tight')

	plt.show()

def TnPolyPlot():
	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))

	axs = axs.flatten()

	nvals = [1,2,3,4]

	params = [[0.5,0],[1.5,0],[0.5,2],[1.5,2]]

	x = np.linspace(-1,1,300)
   
	for nn in nvals:
		for i,pair in enumerate(params):

			b,g = pair
			if i == 0:
				jones = process_latex("$T_{}(x)$".format(nn))
			else:
				jones = None
			tvals = [firewalkW(nn,xx,b,g) for xx in x]
			axs[i].plot(x, tvals, label=jones)
			# axs[i].set_title("\u03b2 = {}; \u03b3 = {}".format(b,g))
			axs[i].set_ylim([-10,10])
			axs[i].set_xlim([-1,1])

			axs[i].axhline(0,color = 'k', linewidth = 0.5)
			axs[i].vlines([0],-10,10,color = 'k',linewidth = 0.5)
			# axs[i].set_xlabel('x')
			axs[i].set_xticks([-1,0,1],[-1,0,1])
			axs[i].set_yticks([-10,0,10],[-10,0,10])
			
	
	axs[0].set_title(process_latex("$\\beta = 0.5$"))
	axs[0].set_ylabel(process_latex("$\\gamma = 0$"),fontsize = 14)
	
	axs[1].set_title(process_latex("$\\beta = 1.5$"))
	axs[2].set_ylabel(process_latex("$\\gamma = 2$"),fontsize = 14)

	handles, labels = axs[0].get_legend_handles_labels()
	# Create a single legend outside the subplots
	leg =  axs[1].legend(handles, labels,loc = 'lower right')
	leg.get_frame().set_linewidth(2)

	# Adjust the layout to make room for the legend
	# fig.tight_layout(rect=[0, 0,0.85, 1])

	# plt.savefig("Tn_polys_4.pdf",bbox_inches = 'tight')

	plt.show()

def TheGamblersPlot():

	beta = 1.1
	gamma = 1
	N = 10
	J = 100

	bd = BDensemble(beta,gamma,N,J,100)

	js,Fs,ts = run_trajectory()

	# compute burn probability for each population

   

	bps = [BurnProb(js[n],J-Fs[n],beta,gamma) for n in tqdm(range(len(js)))]

	fig,ax = plt.subplots(ncols = 2)

	# plot a trajectory of the random process
	ax[0].plot(ts,js,label = process_latex('population $j(t)$'),color = 'orangered')
	ax[0].plot(ts,Fs,label = process_latex('footprint $F(t)$'),color = 'black')

	leg0 = ax[0].legend()
	leg0.get_frame().set_linewidth(2)
	ax[0].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
	ax[0].set_ylabel("number of parcels")
	ax[0].set_xlabel(process_latex('time $(t)$'))

	ax[1].plot(ts,bps)
	ax[1].set_xlabel(process_latex('time $(t)$'))
	ax[1].set_ylabel("Burn probability")
	ax[1].set_ylim(0,1)


	plt.show()
	
def medianPlot():
	plot = MP()
	br = np.linspace(0,4,500)

	# plot.add_plot(br,averageLifetime(1,br,0), label = 'average lifetime',color = 'orangered')

	
	plot.add_plot(br,medianLifetime(50,br,0), label = 'N = 50')
	plot.add_plot(br,medianLifetime(5,br,0), label = 'N = 5',color = 'orange')
	plot.add_plot(br,medianLifetime(1,br,0), label = 'N = 1',color = 'green')

	plot.set_ylabel("Median absorption time")
	plot.set_xlabel(process_latex("Birth rate $\\beta$"))
	plot.show_legend()
	plot.ax.vlines([1],0,100,color = 'black',linestyle = 'dashed')
	plot.ax.set_yscale('log')
	plot.set_ylim(0,80)
	plot.set_xlim(0,4)

	plot.set_xticks([0,1,2,3,4],[0,1,2,3,4])

	
	plot.set_yticks([0.5,1,5,10,50])
	plot.ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

	plot.add_letter_to_corner("B",'northwest')
	
	# plot.save_fig("zerog_medlifes.pdf")
	plt.show()

# Example usage:
if __name__ == "__main__":

	# the Jean curve
	import time

	# # ftpt data generation
	# print(asymAbsorb())


	# N = 5
	# J = 10
	# beta1 = 1.1

	# t1 = 3

	# gv = .5

	# ii = 42
	# print(bigX(t1,beta1))
	# old = [auxiliaryS(ii,j,t1,beta1,gv) for j in range(50)]

	# for i,x in enumerate(old):
	#     print(i,x)
	# s1 = time.time()
	# dist1 = [P(N,n,t1,beta1,gv) for n in range(J)]
	# absorb1 = absorbProb(t1,N,beta1,gv)
	# s2 = time.time()
	# print("total:", absorb1 + sum(dist1))
	# print('time',s2-s1)

	# dist1_vec = P_vectorized([N-1,N],range(J),t1,beta1,gv)

	# s3 = time.time()

	# print("total:", absorb1 + sum(dist1_vec))
	# print('time',s3-s2)



	






