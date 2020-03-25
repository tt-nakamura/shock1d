# reference:
#   I. Danaila, et al.
#    "An Introduction to Scientific Computing" chapter 10

import numpy as np
from scipy.optimize import newton

class Shock1DExact:
    def __init__(self, rho_L, pres_L, rho_R, pres_R, gamma=1.4):
        """ exact solution of one dimensional shock tube
        rho_L, pres_L = density and pressure at x<0, t=0
        rho_R, pres_R = density and pressure at x>0, t=0
        gamma = c_p/c_v: specific heat ratio
        assume rho_L > rho_R and pres_L > pres_R
        """
        global gm,Ms,rL,pL,rR,pR,r1,u1,p1,r2,aL,aR,a2
        gm = gamma
        rL,pL,rR,pR = rho_L, pres_L, rho_R, pres_R
        aL = (gm*pL/rL)**.5 # sound speed at x<0
        aR = (gm*pR/rR)**.5 # sound speed at x>0

        # Mach number of shock wave
        Ms = newton(lambda x: x - 1/x
                    - aL/aR*(gm+1)/(gm-1)
                    *(1 - (pR/pL*(2*gm*x**2 - (gm-1))
                           /(gm+1))**((gm-1)/(2*gm)))
                    ,1)

        # from contact discontinuity to shock wave
        r1 = rR/(2/Ms**2 + gm-1)*(gm+1)
        u1 = 2*aR/(gm+1)*(Ms - 1/Ms)
        p1 = pR*(2*gm*Ms**2 - (gm-1))/(gm+1)

        # from contact discontinuity to expansion fan
        r2 = rL*(p1/pL)**(1/gm)
        a2 = (gm*p1/r2)**.5


    def profile(self, t):
        """ return x,r,u,p = np.array of
        coordinate, density, velocity, pressure
        at time t > 0
        shock wave starts from x=0 at t=0
        solution depends only on x/t
        """
        x1 = -aL*t
        x2 = (u1 - a2)*t
        x3 = u1*t # contact discontinuity
        x4 = Ms*aR*t # shock wave

        # expansion fan
        N = 33
        xE = np.linspace(x1,x2,N)
        uE = 2/(gm+1)*(aL + xE/t)
        rE = rL*(1 - (gm-1)/2*uE/aL)**(2/(gm-1))
        pE = pL*(rE/rL)**gm

        L = 1
        x = np.hstack((x1-L, xE, x3,x3,x4,x4,x4+L))
        r = np.hstack((rL, rE, r2,r1,r1,rR,rR))
        u = np.hstack((0, uE, u1,u1,u1,0,0))
        p = np.hstack((pL, pE, p1,p1,p1,pR,pR))

        return x,r,u,p
