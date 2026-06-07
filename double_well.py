import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

"""
Bulk energy for Ginzburg Landau free energy, described by double well potential
"""

# Global style (LaTeX)  
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color":"#111111",
})
BG = "white"
BLUE = "#1a6faf" # main curve
GOLD = "#c9770a" # minima markers
ROSE = "#b5303a" # barrier / critical
GREY = "#777777" # secondary / annotations
COOL_CM = plt.cm.Blues_r # for varying a panel
WARM_CM = plt.cm.Oranges # for varying b panel

#--------------------------------------------------------------
# Functions
phi = np.linspace(-2.0, 2.0, 800)
# Double well
def f(phi, a, b):
    return 0.5 * a * phi**2 + 0.25 * b * phi**4

def phi_min(a, b):
    return np.sqrt(-a / b)

# -----------------------------------------------------------
# FIGURE 1 – canonical double-well  (a = −1, b = 1)
fig1, ax = plt.subplots(figsize=(6.5, 4.5), facecolor=BG)
ax.set_facecolor(BG)

a0, b0 = -1.0, 1.0
f0 = f(phi, a0, b0)
pm = phi_min(a0, b0)
fm = f(pm, a0, b0)

# main curve
ax.plot(phi, f0, color=BLUE, lw=2.5, zorder=5)

# minima dots
ax.plot([ pm, -pm], [fm, fm], 'o', color=GOLD, ms=8, zorder=6,
        markeredgecolor='white', markeredgewidth=1.2)

print(f'Spinodal point φ = ±sqrt(|a|/b) = ±{pm:.3f}, Binodal point φ = ±sqrt(|a|/3b) = ±{np.sqrt(-a0/(3*b0)):.3f}')
phi_custom = np.sqrt(-a0/(3 *b0))

f_custom = f(phi_custom, a0, b0)
ax.scatter([-phi_custom,phi_custom], [f_custom,f_custom], color=ROSE, s=80, zorder=7,
           edgecolors='white', linewidths=1.2, label=fr'$\phi={phi_custom}$')

# symmetry axis
ax.axvline(0, color=GREY, lw=0.9, ls='--', alpha=0.55)

# arrows from origin to minima
for sign in [+1, -1]:
    ax.annotate('', xy=(sign * pm, fm), xytext=(0, fm), arrowprops=dict(arrowstyle='->', color=GOLD, lw=1.3, connectionstyle='arc3,rad=0'))

# minima labels
x_add, y_add = 0.35, -0.1
fontsize = 14
ax.text( pm - x_add, fm + y_add, r'$\phi_{+}=+\sqrt{|a|/b}$', color=GOLD, fontsize=fontsize, va='bottom')
ax.text(-pm + x_add, fm + y_add, r'$\phi_{-}=-\sqrt{|a|/b}$', color=GOLD, fontsize=fontsize, va='bottom', ha='right')

# barrier arrow & label
ax.annotate('', xy=(0, 0), xytext=(0, fm), arrowprops=dict(arrowstyle='<->', color=ROSE, lw=1.3))
ax.text(0.07, fm * 0.8, r'$\Delta f = \frac{a^2}{4b}$', color=ROSE, fontsize=fontsize, va='center')

# zero line
ax.axhline(0, color=GREY, lw=0.6, ls=':', alpha=0.5)

ax.set_xlabel(r'$\phi$', fontsize=fontsize, labelpad=5)
ax.set_ylabel(r'$f(\phi)$', fontsize=fontsize, rotation=0, labelpad=20)
ax.set_xlim(-1.75, 1.75)
ax.set_ylim(-0.4, 0.3)
ax.set_xticks([]) 
ax.set_yticks([])
ax.set_title(r'Ginzburg-Landau free energy: $f(\phi)=\frac{a}{2}\phi^2+\frac{b}{4}\phi^4$' '\n' r'($a=-1,\;b=1$)', fontsize=fontsize+2, pad=9, loc='left')

fig1.tight_layout()
save_dir = fr""
filename_pdf = str(save_dir) + f'/double_well.pdf'
plt.savefig(filename_pdf, dpi=150, bbox_inches='tight')

# ------------------------------------------------------------
# FIGURE 2 – varying parameters
fig2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.2),
                                   facecolor=BG, sharey=False)

for ax in [ax_a, ax_b]:
    ax.set_facecolor(BG)
    ax.axhline(0, color=GREY, lw=0.6, ls=':', alpha=0.5)

# Left: varying a (b = 1)
a_vals = np.linspace(-1.5, 0.6, 10)
cmap_a = plt.cm.viridis

for i, ai in enumerate(a_vals):
    fi = f(phi, ai, 1.0)
    c  = cmap_a(i / (len(a_vals) - 1))
    ax_a.plot(phi, fi, color=c, lw=1.6, alpha=0.9)

# highlight critical curve a = 0
fc = f(phi, 0, 1.0)
ax_a.plot(phi, fc, color=ROSE, lw=2.2, ls='--',
          label=r'$a=0$ (critical)', zorder=5)

ax_a.set_xlim(-2, 2); ax_a.set_ylim(-0.70, 1.05)
ax_a.set_xlabel(r'$\varphi$', fontsize=11)
ax_a.set_ylabel(r'$f(\varphi)$', fontsize=11, rotation=0, labelpad=18)
ax_a.set_title(r'Varying $a$  ($b=1$ fixed)', fontsize=11, pad=7)
ax_a.legend(fontsize=8.5, frameon=True, framealpha=0.85, edgecolor='#cccccc', loc='upper center')

sm_a = plt.cm.ScalarMappable(cmap=cmap_a, norm=plt.Normalize(a_vals.min(), a_vals.max()))
sm_a.set_array([])
cb_a = fig2.colorbar(sm_a, ax=ax_a, fraction=0.046, pad=0.03)
cb_a.set_label(r'$a$', fontsize=10, rotation=0, labelpad=10)

# Right: varying b (a = −1)
b_vals = np.linspace(0.3, 2.5, 10)
cmap_b = plt.cm.plasma

for i, bi in enumerate(b_vals):
    fi = f(phi, -1.0, bi)
    c  = cmap_b(i / (len(b_vals) - 1))
    ax_b.plot(phi, fi, color=c, lw=1.6, alpha=0.88)

ax_b.set_xlim(-2, 2); ax_b.set_ylim(-1.05, 2.1)
ax_b.set_xlabel(r'$\varphi$', fontsize=11)
ax_b.set_ylabel(r'$f(\varphi)$', fontsize=11, rotation=0, labelpad=18)
ax_b.set_title(r'Varying $b$  ($a=-1$ fixed)', fontsize=11, pad=7)

sm_b = plt.cm.ScalarMappable(cmap=cmap_b, norm=plt.Normalize(b_vals.min(), b_vals.max()))
sm_b.set_array([])
cb_b = fig2.colorbar(sm_b, ax=ax_b, fraction=0.046, pad=0.03)
cb_b.set_label(r'$b$', fontsize=10, rotation=0, labelpad=10)

fig2.tight_layout()

plt.show()
