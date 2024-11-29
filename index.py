import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import seaborn as sns 

fig1, ax1 = plt.subplots(1)
fig2, ax2 = plt.subplots(1)
fig3, ax3 = plt.subplots(1)
fig1.tight_layout()

sns.set_theme( style = "white")

#solve the second order differential equation for the soliton,
def energy_density(x, v, t, w, r):
    return (0.5*v**2 + potential_corr(x, w, r))

def potential(x,w,r): 
    return (1/4)*(x**2-(1-r))**2 + w*(x-1)

def potential_derivative(x,w,r):
    return x*(x**2-(1-r)) + w

def potential_corr(x, E, r):
    x_plus = Newton_Raphson(1, E, r, 100)
    return potential(x, E, r) - potential(x_plus, E, r)

def potential_sec_der(x, w, r):
    return 3*x**2-(1-r) 

def Newton_Raphson(x, w, r, n):
    for i in range(n):
        x += -potential_derivative(x, w, r) / potential_sec_der(x, w, r)
    return x

def ODE(x, v, t, w, r):
    return -2*v/t + potential_derivative(x, w, r)

def RK_22(x0, t0, v0, t_range, x_range, E, r):
    # Boxes to fill:
    xh_total = []
    t_total = []
    v_total = []

    while t0 < t_range:
        xh = x0 + dt * v0 / 2
        if abs(x0) > x_range:
            break

        vh = v0 + ODE(x0, v0, t0, E, r) * dt / 2
        x0 += dt * vh
        v0 += dt * ODE(xh, vh, t0 + dt / 2, E, r)
        t0 += dt
    
        # Fill the boxes:
        v_total.append(vh)
        xh_total.append(xh)
        t_total.append(t0)
        
    return np.array([t_total, xh_total, v_total])

def IntBisec(a_u, a_o, E, r, N):
    for i in range(N):

        Phi_u = RK_22(a_u, t0, v0, t_range, x_range, E, r)
        amid = np.float128(0.5 * (a_u + a_o))

        Phi_mid = RK_22(amid, t0, v0, t_range, x_range, E, r)

        if abs(Phi_u[0, -1] - Phi_mid[0, -1]) < 0.000001:
            a_u = np.float128(amid)
        else:
            a_o = np.float128(amid)
    return amid

# Fundamental values we require:
N = 1
E = np.linspace(0, 0.09, N)
r = 0

# Initial conditions for the Runge-Kutta algorithm.
t0 = 1e-15
v0 = 0
dt = 0.01
t_range = 50

S_Euclidean = []

for i in range(N):    
    
    x_range = 2
    
    a_u = np.float128(-0.1)
    a_o = np.float128(Newton_Raphson(-0.9, E[i], r, 1))
    
    a_mid = IntBisec(a_u, a_o, E[i], r, 100)
        
    phi_mid = RK_22(a_mid, t0, v0, t_range, x_range, E[i], r)
    x_pot = np.linspace(-1.5,1.5,100)

    t = phi_mid[0]
    x = phi_mid[1]
    v = phi_mid[-1]
    
    #search algorithm
    n = 0
    for j in range(len(t)): 
        if abs(t[j] - 30) < 0.000001: 
            n=j
            break
        
    
    t_cut = t[:3500]
    x_cut = x[:3500]
    v_cut = v[:3500]
    
    # Finding the index of intersection:
    #for k in np.arange(0, len(x) - 1):
     #   if np.sign(phi_mid[1, k]) != np.sign(phi_mid[1, k + 1]):
      #      m = round(k)
       #     break

    #R = 0.5 * (phi_mid[0, m] + phi_mid[0, m + 1])
    
        
    dE_dr = energy_density(x_cut, v_cut, t_cut, E[i], r)
    
    ax1.plot(t_cut, x_cut, label="$\epsilon$ = " + str(np.round(E[i], 3))+"$\lambda$")
    ax2.plot(t_cut, dE_dr,  label="$\epsilon$ = " + str(np.round(E[i], 3))+"$\lambda$")
    
    energy = simpson(dE_dr, dx=0.01)
    S_Euclidean.append(energy)
    

def Z(E): 
    return E**(1/4)

def arcsech(x): 
    return np.log(1/x + np.sqrt(1/x**2-1))


#Z_2 kink solution:
ax3.axhline((1-r)**(3/2)*2*np.sqrt(2)/3, label="$Z_2$ kink", linestyle="--", color="k")
ax3.plot(E, (1-r)**(3/2)*2*np.sqrt(2)/3 -(1/6 + np.log(16*np.sqrt(2)/E)*(1-r)**(3/2))*E/np.sqrt(2), label="analytical approx.", linestyle="-.")
ax3.plot(E, S_Euclidean, label="numerical")

"""___________________________________Styles_______________________________"""

density_title = ""
if r==0:
    density_title = "0"
else: 
    density_title = str(r) + "$\mu^2M^2$"

s=11
ax1.set_ylim(-1.1, 1.1)
#ax1.set_xlim(0, 30)
ax1.tick_params(axis='both', which='major', labelsize=s)
ax1.set_title(r"Kinks with several values of $\epsilon$ ( $\rho$ = " + density_title + ' )')
ax1.set_xlabel('$\sqrt{\lambda}\eta$r', size=20)
ax1.set_ylabel('$\phi/\eta$', size=20)

ax2.set_title(r'Energy density for several $\epsilon$ ( $\rho$ = ' + density_title +' )')
ax2.set_xlabel('$\sqrt{\lambda}\eta$x', size=20)
ax2.set_ylabel(r'Energy density/$\sqrt{\lambda}\eta^3$', size=20)
ax2.set_ylim(0, 0.51)


ax3.set_title("Energy of the kinks vs. $\epsilon$")
ax3.set_ylabel("$E_{min}/\sqrt{\lambda}\eta^3$", size=20)
ax3.set_xlabel("$\epsilon/\lambda $", size = 20)

ax1.legend(loc='center right', fontsize=s)
ax2.legend(loc='center right', fontsize=s)
ax3.legend(loc='upper right', fontsize=s)


#always have this: 
plt.show()
