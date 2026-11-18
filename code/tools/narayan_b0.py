# Narayan, Johnson & Gammie 2019 oracles at b = 0 (radial ray, Schwarzschild), j = P/(4 pi), P = 1/[(1-2/r)^(1/2) 4 pi r^4]
# I = int g^4 j ds, ds = proper length in the gas frame = dr/g for a radial E = 1 photon (dr/dlambda = 1, nu_e = 1/g)
from math import pi, sqrt
from scipy.integrate import quad
j = lambda r: 1.0 / (sqrt(1 - 2 / r) * 4 * pi * r**4) / (4 * pi)
static = quad(lambda r: (1 - 2 / r)**1.5 * j(r), 2, float('inf'))[0]       # g = sqrt(1-2/r): g^3 j dr
infall = quad(lambda r: (1 - sqrt(2 / r))**3 * j(r), 2, float('inf'))[0]  # g = 1 - beta, beta = sqrt(2/r)
print(f"static I(0) = {static:.6e} (closed form 1/(1536 pi^2) = {1/(1536*pi*pi):.6e}; paper 6.60e-5)")
print(f"infall I(0) = {infall:.6e} (paper 4.16e-6); ratio {static/infall:.2f}")
