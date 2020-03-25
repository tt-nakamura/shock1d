import numpy as np
import matplotlib.pyplot as plt
from Shock1DExact import Shock1DExact
from Shock1DNumerical import Shock1DNumerical

method = 'Lax-Wendroff'
e = Shock1DExact(1, 1, 0.1, 0.1)
n = Shock1DNumerical(2,method)
x,r,u,p = e.profile(0.4)
n.init(1, 1, 0.1, 0.1)
n.run(0.4)

plt.figure(figsize=(6.4, 8))

plt.subplot(3,1,1)
plt.plot(x,r,'k',label='exact')
plt.plot(n.coordinate(), n.density(),'r.',label=method)
plt.ylabel(r'$\rho$ = density', fontsize=14)
plt.axis([-1,1,0,1.1])
plt.legend(loc='upper right')

plt.subplot(3,1,2)
plt.plot(x,u,'k',label='exact')
plt.plot(n.coordinate(), n.velocity(),'r.',label=method)
plt.ylabel(r'$u$ = velocity', fontsize=14)
plt.axis([-1,1,0,1.5])
plt.legend(loc='upper left')

plt.subplot(3,1,3)
plt.plot(x,p,'k',label='exact')
plt.plot(n.coordinate(), n.pressure(),'r.',label=method)
plt.ylabel(r'$p$ = pressure', fontsize=14)
plt.axis([-1,1,0,1.1])
plt.legend(loc='upper left')

plt.xlabel('x', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('fig2.pdf')
plt.show()
