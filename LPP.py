import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hyp2f1, gamma

def LPP(phi, kappa, F):
    A = gamma(4/kappa) / (np.sqrt(np.pi) * gamma((8 - kappa)/(2 * kappa)))
    return 0.5 + A * 1/np.tan(phi) * F(0.5, 4/kappa, 1.5, -1/np.tan(phi)**2)

kappa = [1, 2, 3, 4, 5, 6, 7, 8]
phi = np.linspace(1e-6, np.pi-1e-6, 300)

plt.figure(figsize=(8, 6))

for k in kappa:
    LPP_kappa = LPP(phi, k, hyp2f1)
    plt.plot(phi, LPP_kappa, label=f"$\\kappa = {k}$")
plt.xlabel("Polar Angle")
plt.ylabel("Left Passage Probability")
plt.legend()
plt.xlim([0, np.pi])
plt.ylim([0, 1])
plt.grid(alpha=0.4)
xticklabels = ["0", r"$\pi/2$", r"$\pi$"]
plt.xticks([0, np.pi / 2, np.pi], labels=xticklabels)
plt.tight_layout()
plt.show()

