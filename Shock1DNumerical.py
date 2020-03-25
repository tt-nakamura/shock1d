# reference:
#   I. Danaila, et al.
#    "An Introduction to Scientific Computing" chapter 10

import numpy as np

class Shock1DNumerical:
    def __init__(self, length, method='LW',
                 ngrid=256, gamma=1.4, visc=0, CFL=0.95):
        """ simulation of one dimensional shock tube
        length = tube length; origin is at tube's midpoint
        method = LW (Lax-Wendroff) or Mac (MacCormack) or Roe
        ngrid = number of mesh points
        gamma = c_p/c_v: ratio of specific heat
        visc = artificial viscosity (for LW and Mac only)
        CFL = Courant-Friedrichs-Lewy condition < 1
        """
        global N,x,dx,gm,C,D,W,integrator
        N,gm = ngrid, gamma
        dx = length/N
        C,D = CFL*dx, visc*dx
        x = np.linspace(-length/2, length/2, N+1)

        # mass, momentum, energy density
        W = np.zeros((3,N+1))

        m = method[0].lower()
        if   m == 'l': integrator = LaxWendroff
        elif m == 'm': integrator = MacCormack
        elif m == 'r': integrator = Roe
        else: raise RuntimeError('unknown method')

    # origin of x is at midpoint in [-L/2, L/2]
    def coordinate(self): return x
    def density(self): return density()
    def velocity(self): return velocity()
    def pressure(self): return pressure()

    def init(self,rL,pL,rR,pR):
        """ set initial condition
        rL,pL = density and pressure at x<0, t=0
        rR,pR = density and pressure at x>0, t=0
        """
        N2 = N//2
        W[0,N2:] = rR
        W[1,N2:] = 0
        W[2,N2:] = pR/(gm-1)
        W[0,:N2] = rL
        W[1,:N2] = 0
        W[2,:N2] = pL/(gm-1)

    def run(self,t):
        """ run simulation for time t until
        shock wave reaches tube's edge
        """
        while t>0:
            dt = time_step()
            if t<dt: dt=t
            integrator(dt)
            t -= dt

def density(): return W[0]
def velocity(): return W[1]/W[0]
def pressure(): return (gm-1)*(W[2] - W[1]**2/W[0]/2)
def sound_speed(): return np.sqrt(gm*pressure()/W[0])
def enthalpy(): return gm*W[2]/W[0] - (gm-1)*(W[1]/W[0])**2/2

def time_step():
    """ CFL condition """
    u = np.abs(W[1]/W[0])
    dt = C/np.max(u + sound_speed())
    if integrator == Roe: dt/=2
    return dt

def flux(W): 
    F = np.empty_like(W)
    u = W[1]/W[0] # velocity
    KE = W[1]*u/2 # kinetic energy density
    F[0] = W[1] # mass flux
    F[1] = (gm-1)*W[2] + (3-gm)*KE # momentum flux
    F[2] = (gm*W[2] - (gm-1)*KE)*u # energy flux
    return F

def LaxWendroff(dt):
    F = flux(W)
    F[:,1:] -= D*np.diff(W) # artificial viscosity
    # predictor
    W1 = (W[:,:-1] + W[:,1:])/2 - np.diff(F)/dx*dt/2
    F = flux(W1)
    F[:,:-1] -= D*np.diff(W1)
    # corrector
    W[:,1:-1] -= np.diff(F)/dx*dt

def MacCormack(dt):
    F = flux(W)
    F[:,1:] -= D*np.diff(W) # artificial viscosity
    # predictor
    W1 = W[:,:-1] - np.diff(F)/dx*dt
    F = flux(W1)
    F[:,:-1] -= D*np.diff(W1)
    # corrector
    W[:,1:-1] = (W[:,1:-1] + W1[:,1:])/2 - np.diff(F)/dx*dt/2

def Roe(dt):
    rho = density()
    u = velocity()
    h = enthalpy()
    r = np.sqrt(rho[1:]/rho[:-1])
    u = (r*u[1:] + u[:-1])/(r+1) # Roe average
    h = (r*h[1:] + h[:-1])/(r+1)
    a = np.sqrt((gm-1)*(h - u**2/2))
    b1 = (gm-1)*(u/a)**2/2
    b2 = (gm-1)/a**2
    
    # S = sign of eigen values
    S = np.array([np.sign(u-a),
                  np.sign(u),
                  np.sign(u+a)])
    # P = eigen vectors in columns
    P = np.array([np.ones((3,N)),
                  [u-a, u, u+a],
                  [h-a*u, u**2/2, h+a*u]])
    # Q = inverse matrix of P
    Q = np.array([[(b1 + u/a)/2, -(b2*u + 1/a)/2, b2/2],
                  [1-b1, b2*u, -b2],
                  [(b1 - u/a)/2, -(b2*u - 1/a)/2, b2/2]])
    F = flux(W)
    D = np.diff(F)
    D = np.einsum('ijk,jk->ik',Q,D)
    D = np.einsum('ijk,jk->ik',P,S*D)
    F = (F[:,:-1] + F[:,1:] - D)/2
    W[:,1:-1] -= np.diff(F)/dx*dt
