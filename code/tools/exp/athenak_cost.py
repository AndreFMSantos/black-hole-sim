# estimated cost of the stock run (MAD 8_4) from the measured Mzcps
mz_small, mz_big = 3.19e6, 8.35e6          # measured (double), 147k and 1.18M cells
blocks, nb = 4096, 8                       # original input: 8^3 blocks
cells = blocks*nb**3
ghost = (nb+8)**3*blocks                   # nghost=4 -> 16^3 per block
dx_fine = 64.0/512                         # 2048/32 = 64 at level 0; 9 levels
dt = 0.3*dx_fine                           # CFL of the measured case: dt=1.2 with dx=4
for T in (1e4, 2e4):
    cyc = T/dt
    zc = cyc*cells
    print(f"T={T:.0f} M: cycles={cyc:.3g} zone-cycles={zc:.3g}  double: {zc/mz_big/3600:.1f} h  (at 3.19 Mzcps: {zc/mz_small/3600:.0f} h)")
print("physical cells",cells,"with ghosts",ghost, "ratio", ghost/cells)
