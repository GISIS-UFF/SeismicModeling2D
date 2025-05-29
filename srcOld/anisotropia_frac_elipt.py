import numpy as np
import matplotlib.pyplot as plt

delta = [0 , 0.2 , 0.5, 1]
epsilon = [0 , 0.2 , 0.5, 1]
vpz = 3000
theta = np.linspace(0, 2*np.pi, 1000)  

for i, d in enumerate(delta):
    fig, axs = plt.subplots(2,2,subplot_kw={'projection': 'polar'})
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    axs = axs.flatten()

    for j, e in enumerate(epsilon):
        ax=axs[j]
        
        vp_elip = vpz * np.sqrt(1 + 2 * e * (np.sin(theta)**2))
        vp_frac = vpz * (1 + d * (np.sin(theta)**2) * (np.cos(theta)**2) + e * (np.sin(theta)**4))
        
        ax.plot(theta, vp_elip,'b-', label='Vp_elip')
        ax.plot(theta, vp_frac,'r--', label='Vp_frac')
        
        ax.set_title(f'δ = {d}, ε = {e}', fontsize=10, pad = 25)
        ax.set_yticklabels([])
        ax.set_theta_zero_location('N')  
        ax.grid(True)
        ax.legend(loc = 'best', fontsize = 6)
    

plt.show()