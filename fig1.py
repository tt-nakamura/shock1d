import numpy as np
import matplotlib.pyplot as plt
from Shock1DExact import Shock1DExact

s = Shock1DExact(1, 1, 0.1, 0.1)
x,r,u,p = s.profile(0.4)

plt.figure(figsize=(6.4, 8))

plt.subplot(3,1,1)
plt.plot(x,r,'k')
plt.ylabel(r'$\rho$ = density', fontsize=14)
plt.axis([-1,1,0,1.1])

plt.subplot(3,1,2)
plt.plot(x,u,'k')
plt.ylabel(r'$u$ = velocity', fontsize=14)
plt.axis([-1,1,0,1.1])

plt.subplot(3,1,3)
plt.plot(x,p,'k')
plt.ylabel(r'$p$ = pressure', fontsize=14)
plt.axis([-1,1,0,1.1])

plt.xlabel('x', fontsize=14)
plt.tight_layout()
plt.savefig('fig1.pdf')
plt.show()
