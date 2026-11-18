# Image catalogue (assets/)

`regen_assets.py` reproducibly regenerates the PNG/JPG files in `01-Projects/black-hole-sim/assets/` (not the .mp4 files).
Each entry has the exact commands (apps from `code/build/apps`, RTX 5080 GPU) and the montages (Pillow).

```
python3 regen_assets.py --list                 # entries and the non-reproducible ones (with the reason)
python3 regen_assets.py                        # everything -> $BHS_DATA/assets_new/
python3 regen_assets.py --only v09b-comparison,v21e-kerr-newman-disk --out /tmp/x
```

Notes learned while reproducing (checked against the old images):
- v0_6-kerr-cpu CPU renders use `--exposure 3` (the CPU default is 1.6, the GPU's is 3); without it the disks come out different.
- The 960x540 panels (v09a/v09a2/v09b/v09c) are 32-sample 1080p renders downscaled with Pillow LANCZOS (not a direct render at 960x540).
- GPU: `--prec float` (the default in `--shot` is double and is slow), `--runs 1 --nohud`.
- JPG: Pillow, 4:2:0, quality 90 (v09c*, v22*) or 88 (v05b-sky-*, best fit, not exact).
- v22f-grmhd: the "17 degrees const" panel needs `--electrons const --ti-te 3`.
- v21d Ellis: fov 50 (not 60); the close Interstellar panel (l = 2.2) uses fov 60.
- v07c-fast-beta07: theta 80 (default), r = 40, `--beta 0,0,0.7` (theta found by sweeping against the old image).

"Non-reproducible" entries (reason in the file itself): crops without a documented position (v05b-filter-crops, v07d-aa-crop, v08c-crops,
v07b-orders-*), video frames (v08b-flight-frame1..4), v22f-torus-oracle, and the dropped v24/v26.
Entries whose result is not identical to the old image even with the command from the notes: v09c-turbulence, v09c2-slowlight (see `note`).
