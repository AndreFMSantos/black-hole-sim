#!/usr/bin/env python3
"""Reproducible catalogue of the images (PNG/JPG) in 01-Projects/black-hole-sim/assets/.

Usage:  python3 regen_assets.py [--only NAME[,NAME...]] [--out DIR] [--list] [--keep-tmp]

Each ENTRIES item is {name, steps[...]} (or {name, reproducible: False, reason}). A step is:
  {"run": "<shell command>"}                      runs an app (placeholders below)
  {"resize": [src, dst, w, h]}                    Pillow LANCZOS (what reproduces the *-960 panels and the montages from 1080p)
  {"grid": [[files...], cols, tw, th, dst]}   pastes tw x th panels into a grid (order: row by row)
  {"jpg": [src, dst, quality]}                    Pillow, subsampling 4:2:0
  {"copy": [src, dst]}
  {"crop": [src, dst, x, y, w, h, scale]}        enlarged crop by nearest neighbour
Paths inside the steps: relative to {T} (the entry's temporary folder) or {OUT} (the output folder).
Placeholders: {G} v3_1-kerr-cuda-ui, {C} v0_6-kerr-cpu, {V3} {V4} {V5} {V7B} the remaining apps, {SKY} {SKY2} {BGM} {OUT} {T}.
"reproducible": False entries are NOT generated (the old image in assets/ stays intact); the reason is recorded.
All renders are deterministic (fixed seed); GPU: RTX 5080, CUDA; the binaries come from code/build (cmake --build code/build).
Needs Pillow (if missing, the script re-invokes itself with `uv run --with pillow --with numpy python -I`).
"""
import os as _os_bhs, pathlib as _pl_bhs
import os
_BHS_ROOT = str(_pl_bhs.Path(__file__).resolve().parents[3])  # the folder holding code/, data/ and assets/
_BHS_DATA = _os_bhs.environ.get('BHS_DATA', _os_bhs.path.join(_BHS_ROOT, 'data'))
import argparse, os, shlex, shutil, subprocess, sys, time

try:
    from PIL import Image
except ImportError:  # re-invoke with uv
    os.execvp("uv", ["uv", "run", "--quiet", "--with", "pillow", "--with", "numpy", "python", "-I", os.path.abspath(__file__)] + sys.argv[1:])

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", ".."))
ASSETS_OLD = os.path.abspath(os.path.join(CODE, "..", "assets"))
APPS = os.path.join(CODE, "build", "apps")
VARS = dict(
    G=f"{APPS}/v3_1-kerr-cuda-ui/v3_1-kerr-cuda-ui", C=f"{APPS}/v0_6-kerr-cpu/v0_6-kerr-cpu",
    V3=f"{APPS}/v0_3-raytrace-cpu/v0_3-raytrace-cpu", V4=f"{APPS}/v0_4-accretion-disk/v0_4-accretion-disk",
    V5=f"{APPS}/v0_5-realtime-gpu/v0_5-realtime-gpu", V7B=f"{APPS}/v0_7b-photon-rings/v0_7b-photon-rings",
    SKY=f"{ASSETS_OLD}/sky/starmap_2020_8k.exr", SKY2=os.path.join(_BHS_ROOT, "assets/sky/wise/wise_8k.exr"),
    BGM=os.path.join(_BHS_DATA, "bgm"), DEFOUT=os.path.join(_BHS_DATA, "assets_new"),
)
GS = "{G} --runs 1 --nohud --prec float"        # GPU, 1 run, float
HPX = "--tonemap aces --bloom 0.1"

ENTRIES = []
def add(name, steps=None, **kw):
    ENTRIES.append(dict(name=name, steps=steps or [], **kw))
def skip_entry(name, reason):
    ENTRIES.append(dict(name=name, reproducible=False, reason=reason, steps=[]))

# ---------- v0.3, v0.4 (CPU, 1920x1080, SSAA 3, procedural sky) ----------
add("v03-shadow.png", [{"run": "{V3} --w 1920 --h 1080 --ssaa 3 --out {OUT}/v03-shadow.png"}])
add("v03-flat-reference.png", [{"run": "{V3} --w 1920 --h 1080 --ssaa 3 --nobh --out {OUT}/v03-flat-reference.png"}])
add("v04-disk.png", [{"run": "{V4} --w 1920 --h 1080 --ssaa 3 --out {OUT}/v04-disk.png"}])                       # theta 80 = 10 degrees above the plane
add("v04-disk-edgeon.png", [{"run": "{V4} --w 1920 --h 1080 --ssaa 3 --theta 86 --out {OUT}/v04-disk-edgeon.png"}])  # 4 degrees
add("v04-disk-nodoppler.png", [{"run": "{V4} --w 1920 --h 1080 --ssaa 3 --nodoppler --out {OUT}/v04-disk-nodoppler.png"}])

# ---------- v0.5 (GLSL GPU), v0.5b real sky ----------
add("v05-gpu.png", [{"run": "{V5} --w 1920 --h 1080 --nosky --shot {OUT}/v05-gpu.png"}])
add("v05-gpu-azimuth.png", [{"run": "{V5} --w 1920 --h 1080 --nosky --theta 65 --phi 40 --dist 25 --shot {OUT}/v05-gpu-azimuth.png"}])
for n, extra in (("theta80", ""), ("edgeon", "--theta 86"), ("einstein-ring", "--theta 61.064 --phi 86.405")):  # vista C = notes/physics/05b
    add(f"v05b-sky-{n}.png", [{"run": "{V5} --w 1920 --h 1080 " + extra + " --shot {OUT}/v05b-sky-" + n + ".png"}])
    add(f"v05b-sky-{n}.jpg", [{"jpg": [f"{{OUT}}/v05b-sky-{n}.png", f"{{OUT}}/v05b-sky-{n}.jpg", 88]}], dep=[f"v05b-sky-{n}.png"])
skip_entry("v05b-filter-crops.jpg", "4x crops (CPU 4x4 / filtered GPU / unfiltered GPU) next to the shadow of view C: the crop position and the enlargement method are not documented")

# ---------- v0.6a Kerr CPU (SSAA 3, real sky, exposure 3, Newtonian profile) ----------
CPU = "{C} --w 1920 --h 1080 --ssaa 3 --sky {SKY} --exposure 3"
add("v06-kerr-a09.png", [{"run": CPU + " --a 0.9 --theta 80 --profile newtonian --out {OUT}/v06-kerr-a09.png"}])
add("v06-kerr-a0998-edgeon.png", [{"run": CPU + " --a 0.998 --theta 86 --profile newtonian --out {OUT}/v06-kerr-a0998-edgeon.png"}])
add("v06-kerr-shadow-a0998.png", [{"run": CPU + " --a 0.998 --theta 90 --nodisk --profile newtonian --out {OUT}/v06-kerr-shadow-a0998.png"}])
add("v06-kerr-shadow-a0.png", [{"run": CPU + " --a 0 --theta 90 --nodisk --profile newtonian --out {OUT}/v06-kerr-shadow-a0.png"}])
# ---------- v0.6b Kerr CUDA (double, Newtonian profile as it was then) ----------
add("v06b-kerr-a09.png", [{"run": GS.replace("float", "double") + " --shot {OUT}/v06b-kerr-a09.png --a 0.9 --theta 80 --profile newtonian"}])
add("v06b-kerr-a0998-edgeon.png", [{"run": GS.replace("float", "double") + " --shot {OUT}/v06b-kerr-a0998-edgeon.png --a 0.998 --theta 86 --profile newtonian"}])
add("v06b-kerr-shadow-a0998.png", [{"run": GS.replace("float", "mixed") + " --tol 1e-6 --shot {OUT}/v06b-kerr-shadow-a0998.png --a 0.998 --theta 90 --nodisk"}])

# ---------- v0.7a/c CPU (NT by default) ----------
add("v07a-nt-a09.png", [{"run": CPU + " --a 0.9 --theta 80 --profile nt --out {OUT}/v07a-nt-a09.png"}])
add("v07c-static-r12.png", [{"run": CPU + " --a 0.9 --theta 85 --dist 12 --observer static --out {OUT}/v07c-static-r12.png"}])
add("v07c-orbit-r12.png", [{"run": CPU + " --a 0.9 --theta 85 --dist 12 --observer orbit --out {OUT}/v07c-orbit-r12.png"}])
# theta 80 (default): the note only gives beta = 0.7 along +phi and r = 40; theta confirmed by a sweep (75..90) against the old image
add("v07c-fast-beta07.png", [{"run": CPU + " --a 0.9 --theta 80 --dist 40 --beta 0,0,0.7 --out {OUT}/v07c-fast-beta07.png"}])
add("v07c-infall-r10.png", [{"run": CPU + " --a 0.9 --theta 85 --dist 10 --observer infall --out {OUT}/v07c-infall-r10.png"}])

# ---------- v0.7b rings by order (CPU) ----------
skip_entry("v07b-orders-overview.png", "command not documented (only fov 20); no confirmation of --theta/--dist/--ssaa and of the rest of the scene")
skip_entry("v07b-orders-zoom.png", "the zoom centre (--lookx) for fov 0.12 is not documented")

# ---------- v0.7d GPU (64 samples, float) ----------
G64 = GS + " --samples 64 --shot "
CIN = "--tonemap aces --bloom 0.12 --beaming 0.5 --clamp 50"   # v0.7d cinematic preset (without the v0.8d extras)
add("v07d-a09-reinhard.png", [{"run": G64 + "{OUT}/v07d-a09-reinhard.png --a 0.9 --theta 80"}])
add("v07d-64samples.png", [{"run": G64 + "{OUT}/v07d-64samples.png --a 0.9 --theta 80"}])      # == a09-reinhard
add("v07d-1sample.png", [{"run": GS + " --samples 1 --shot {OUT}/v07d-1sample.png --a 0.9 --theta 80"}])
add("v07d-a09-cinematic.png", [{"run": G64 + "{OUT}/v07d-a09-cinematic.png --a 0.9 --theta 80 " + CIN}])
add("v07d-a0998-reinhard.png", [{"run": G64 + "{OUT}/v07d-a0998-reinhard.png --a 0.998 --theta 86"}])
add("v07d-a0998-cinematic.png", [{"run": G64 + "{OUT}/v07d-a0998-cinematic.png --a 0.998 --theta 86 " + CIN}])
add("v07d-orbit-r12.png", [{"run": G64 + "{OUT}/v07d-orbit-r12.png --a 0.9 --theta 85 --dist 12 --observer orbit"}])
add("v07d-orbit-r12-cinematic.png", [{"run": G64 + "{OUT}/v07d-orbit-r12-cinematic.png --a 0.9 --theta 85 --dist 12 --observer orbit " + CIN}])
skip_entry("v07d-aa-crop.png", "3x crop of 1 vs 64 samples: the crop position is not documented")

# ---------- v0.8 ----------
skip_entry("v08b-flight-frame1.png", "frame extracted from the flight clip (mp4, out of scope): the extracted instants are not documented")
skip_entry("v08b-flight-frame2.png", "same")
skip_entry("v08b-flight-frame3.png", "same")
skip_entry("v08b-flight-frame4.png", "same")
skip_entry("v08c-crops.png", "crops (N=1/N=4, with and without OptiX, 256 reference) from undocumented positions; and OptiX is not compiled in code/build")
NOPOL = "--cinematic --glare 0 --grain 0 --vignette 0 --chroma 0"
add("v08d-a09-no-polish.png", [{"run": G64 + "{OUT}/v08d-a09-no-polish.png --a 0.9 --theta 80 " + NOPOL}])
add("v08d-a09-polish.png", [{"run": G64 + "{OUT}/v08d-a09-polish.png --a 0.9 --theta 80 --cinematic"}])
add("v08d-a0998-no-polish.png", [{"run": G64 + "{OUT}/v08d-a0998-no-polish.png --a 0.998 --theta 86 " + NOPOL}])
add("v08d-a0998-polish.png", [{"run": G64 + "{OUT}/v08d-a0998-polish.png --a 0.998 --theta 86 --cinematic"}])

# ---------- v0.9a/a2/b/c/c2: panels = 1920x1080 render of 32 samples downscaled with Pillow LANCZOS ----------
def panel(name, args, w=960, h=540, samples=32, extra_dir="T"):
    """1080p render + LANCZOS -> w x h; returns the steps (the large PNG stays in {T})."""
    big = "{T}/" + name.replace("/", "_") + "_big.png"
    return [{"run": GS + f" --samples {samples} --shot {big} --a 0.9 " + args},
            {"resize": [big, "{OUT}/" + name, w, h]}]
V9A = "--theta 75"
add("v09a/base-960.png", panel("v09a/base-960.png", V9A))
add("v09a/fcol-960.png", panel("v09a/fcol-960.png", V9A + " --fcol 1.7"))
add("v09a/limb-960.png", panel("v09a/limb-960.png", V9A + " --limb eddington"))
add("v09a/both-960.png", panel("v09a/both-960.png", V9A + " --fcol 1.7 --limb eddington --expose physical"))
add("v09a-comparison.png", [{"grid": [[f"{{OUT}}/v09a/{n}-960.png" for n in ("base", "fcol", "limb", "both")], 2, 960, 540, "{OUT}/v09a-comparison.png"]}],
    dep=[f"v09a/{n}-960.png" for n in ("base", "fcol", "limb", "both")])
add("v09a2/hard-960.png", panel("v09a2/hard-960.png", V9A + " --rout 16"))
add("v09a2/soft-960.png", panel("v09a2/soft-960.png", V9A + " --edge-soft 0.12"))
add("v09a2/soft-noise-960.png", panel("v09a2/soft-noise-960.png", V9A + " --edge-soft 0.12 --edge-noise 0.1"))
add("v09a2/wide-960.png", panel("v09a2/wide-960.png", V9A + " --rout 22 --edge-soft 0.2 --edge-noise 0.12"))
add("v09a2-comparison.png", [{"grid": [[f"{{OUT}}/v09a2/{n}-960.png" for n in ("hard", "soft", "soft-noise", "wide")], 2, 960, 540, "{OUT}/v09a2-comparison.png"]}],
    dep=[f"v09a2/{n}-960.png" for n in ("hard", "soft", "soft-noise", "wide")])
b9 = [("thin", "--theta 75"), ("hr05", "--theta 75 --hr 0.05 --tau 10"),
      ("hr10", "--theta 75 --hr 0.1 --flare 0.25 --tau 6 --edge-soft 0.12 --edge-noise 0.1"),
      ("th86", "--theta 86 --hr 0.1 --flare 0.25 --tau 6 --edge-soft 0.12")]
steps = []
for k, a in b9:
    steps += panel(f"_v09b_{k}.png", a)[:1] + [{"resize": ["{T}/_v09b_%s.png_big.png" % k, "{T}/p_%s.png" % k, 960, 540]}]
steps.append({"grid": [[f"{{T}}/p_{k}.png" for k, _ in b9], 2, 960, 540, "{OUT}/v09b-comparison.png"]})
add("v09b-comparison.png", steps)
HR9C = "--theta 75 --hr 0.08 --flare 0.15 --tau 6 --edge-soft 0.12"
c9 = [("smooth", HR9C), ("turb_t0", HR9C + " --turb 0.7 --time 0"), ("turb_t60", HR9C + " --turb 0.7 --time 60"),
      ("spot_t0", HR9C + " --turb 0.7 --spot 8,200,1.2,5 --time 0"), ("spot_t40", HR9C + " --turb 0.7 --spot 8,200,1.2,5 --time 40"),
      ("spot_t80", HR9C + " --turb 0.7 --spot 8,200,1.2,5 --time 80")]
steps = []
for k, a in c9:
    steps += [{"run": GS + f" --samples 32 --shot {{T}}/c9_{k}_big.png --a 0.9 " + a}, {"resize": [f"{{T}}/c9_{k}_big.png", f"{{T}}/c9_{k}.png", 960, 540]}]
steps += [{"grid": [[f"{{T}}/c9_{k}.png" for k, _ in c9], 2, 960, 540, "{T}/v09c-comparison_pre.png"]},
           {"jpg": ["{T}/v09c-comparison_pre.png", "{OUT}/v09c-comparison.jpg", 90]}]
add("v09c-comparison.jpg", steps)
steps = []
for k, a in (("a", HR9C + " --turb 1.2 --turb-scale 1.5"), ("b", HR9C + " --turb 1.6 --turb-scale 1 --tau 3")):
    steps += [{"run": GS + f" --samples 32 --shot {{T}}/t9_{k}_big.png --a 0.9 " + a}, {"resize": [f"{{T}}/t9_{k}_big.png", f"{{T}}/t9_{k}.png", 720, 405]}]
steps += [{"grid": [["{T}/t9_a.png", "{T}/t9_b.png"], 2, 720, 405, "{T}/v09c-turbulence_pre.png"]},
           {"jpg": ["{T}/v09c-turbulence_pre.png", "{OUT}/v09c-turbulence.jpg", 90]}]
add("v09c-turbulence.jpg", steps, note="only the --turb/--turb-scale/--tau parameters are in the notes; the base (HR9C) is assumed: MAE ~1.2 vs the old one, not equal")
S2 = "--theta 80 --hr 0.05 --tau 3 --turb 1.0 --spot 8,60,1.0,12 --time 100"
steps = []
for k, a in (("a", ""), ("b", " --slowlight")):
    steps += [{"run": GS + f" --samples 16 --shot {{T}}/s2_{k}_big.png --a 0.9 " + S2 + a}, {"resize": [f"{{T}}/s2_{k}_big.png", f"{{T}}/s2_{k}.png", 640, 360]}]
steps += [{"grid": [["{T}/s2_a.png", "{T}/s2_b.png"], 2, 640, 360, "{T}/v09c2-slowlight_pre.png"]},
           {"jpg": ["{T}/v09c2-slowlight_pre.png", "{OUT}/v09c2-slowlight.jpg", 90]}]
add("v09c2-slowlight.jpg", steps, note="command from the notes, but the result differs ~1.4 (JPEG q90) from the old one: the old one's outer edge is softer; missing parameters not identified")

# ---------- v2.1 variants (GPU, double) ----------
def grid4(name, panels, w, h, samples, common):
    steps = []
    for k, a in panels:
        steps += [{"run": "{G} --runs 1 --nohud --prec double --tol 1e-8 --samples %d --w %d --h %d --shot {T}/%s_%s.png %s %s" % (samples, w, h, name, k, common, a)}]
    steps.append({"grid": [[f"{{T}}/{name}_{k}.png" for k, _ in panels], 2, w, h, "{OUT}/" + name + ".png"]})
    return steps
add("v21-reissner-nordstrom-gpu.png", grid4("v21-reissner-nordstrom-gpu", [("q0", "--charge 0"), ("q06", "--charge 0.6"), ("q09", "--charge 0.9"), ("q11", "--charge 1.1")], 960, 540, 8,
    "--a 0 --theta 78 --dist 30 --fov 34 --rin 6 --rout 14 --profile newtonian --nodoppler " + HPX + " --exposure 2"))
add("v21c-schwarzschild-de-sitter-gpu.png", grid4("v21c-schwarzschild-de-sitter-gpu", [("l0", "--lambda 0"), ("l001", "--lambda 0.01"), ("l003", "--lambda 0.03"), ("l0045", "--lambda 0.045")], 960, 540, 6,
    "--a 0 --theta 90 --dist 5.5 --fov 70 --nodisk --tonemap aces --exposure 2"))
add("v21e-kerr-newman-disk.png", grid4("v21e-kerr-newman-disk", [("a05q0", "--a 0.5 --charge 0"), ("a05q05", "--a 0.5 --charge 0.5"), ("a05q075", "--a 0.5 --charge 0.75"), ("a09q04", "--a 0.9 --charge 0.4")], 960, 540, 8,
    "--theta 78 --dist 40 --fov 34 --rout 18 --tonemap aces --bloom 0.1 --exposure 3"))
WH = "--wormhole %s --theta 90 --phi 0 --sky2 {SKY2} --tonemap aces --exposure 2 --sky-exposure 3 --samples 6 --prec float --runs 1 --nohud"
add("v21d-wormhole-interstellar.png", [{"run": "{G} " + WH % "interstellar" + " --wh-l 4 --fov 50 --w 1280 --h 720 --shot {OUT}/v21d-wormhole-interstellar.png"}])
add("v21d-wormhole-ellis-vs-near.png", [
    {"run": "{G} " + WH % "ellis,1" + " --wh-l 4 --fov 50 --w 1280 --h 720 --shot {T}/ellis.png"},
    {"run": "{G} " + WH % "interstellar" + " --wh-l 2.2 --fov 60 --w 1280 --h 720 --shot {T}/near.png"},
    {"grid": [["{T}/ellis.png", "{T}/near.png"], 2, 1280, 720, "{OUT}/v21d-wormhole-ellis-vs-near.png"]}])

# ---------- v2.2 radio / GRMHD ----------
RAD = (GS + " --w 700 --h 700 --samples 8 --a 0.9 --dist 150 --fov 14 --disk-model torus --torus 7,12 --ne0 3e4 --plasma-beta 3 --theta-e-max 20 "
       "--mbh 6.5e9 --nu-ghz 230 --tb-range 8,11 --tonemap aces --shot ")
steps = [{"run": RAD + f"{{T}}/r{th}.png --theta {th}"} for th in (17, 60, 80)]
steps += [{"grid": [[f"{{T}}/r{th}.png" for th in (17, 60, 80)], 3, 700, 700, "{T}/v22b-radio_pre.png"]}, {"jpg": ["{T}/v22b-radio_pre.png", "{OUT}/v22b-radio.jpg", 90]}]
add("v22b-radio.jpg", steps)
GRM = (GS + " --w 700 --h 700 --samples 1 --dist 150 --fov 14 --disk-model grmhd --grmhd {BGM}/MAD_a0_5500.bgm --munit 1e26 --mbh 6.5e9 "
       "--nu-ghz 230 --tb-range 8,11 --tonemap aces --grmhd-step 0.03 --shot ")
steps = [{"run": GRM + "{T}/g1.png --theta 17 --electrons rhigh --rhigh 20"}, {"run": GRM + "{T}/g2.png --theta 60 --electrons rhigh --rhigh 20"},
          {"run": GRM + "{T}/g3.png --theta 17 --electrons const --ti-te 3"},
          {"grid": [["{T}/g1.png", "{T}/g2.png", "{T}/g3.png"], 3, 700, 700, "{T}/v22f-grmhd_pre.png"]}, {"jpg": ["{T}/v22f-grmhd_pre.png", "{OUT}/v22f-grmhd.jpg", 90]}]
add("v22f-grmhd.jpg", steps)
skip_entry("v22f-torus-oracle.jpg", "torus and synthetic-dump (torus_a0_synth.bgm) parameters not documented for this panel (analytic / reader / difference x4)")
OWN = ("{G} --nohud --runs 1 --shot {OUT}/v22g-own-mad-th%d.png --grmhd {BGM}/../athenak/run_d/bgm/mad_00200.bgm --disk-model grmhd --electrons rhigh --rhigh 20 "
       "--dist 150 --fov 16 --w 1000 --h 1000 --prec float --samples 8 --theta %d --munit 7.361e24 --emission leung --tb-range 8.5,11")
add("v22g-own-mad-th17.png", [{"run": OWN % (17, 17)}])
add("v22g-own-mad-th60.png", [{"run": OWN % (60, 60)}])

# ---------- dropped (no physical basis; to be deleted) ----------
skip_entry("v24-disk-tex-interstellar.png", "DROPPED (no physical basis): do not regenerate")
skip_entry("v26-disk-arms-rings.png", "DROPPED (no physical basis): do not regenerate")

# ===================== execution =====================
def fmt(s, T, OUT):
    d = dict(VARS); d.update(T=T, OUT=OUT)
    return s.format(**d)

def do_step(p, T, OUT):
    (k, v), = p.items()
    f = lambda s: fmt(s, T, OUT)
    if k == "run":
        cmd = f(v)
        print("  $", cmd, flush=True)
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"exit {r.returncode}: {(r.stderr or r.stdout).strip()[-600:]}")
    elif k == "resize":
        src, dst, w, h = v
        os.makedirs(os.path.dirname(f(dst)), exist_ok=True)
        Image.open(f(src)).convert("RGB").resize((w, h), Image.LANCZOS).save(f(dst))
    elif k == "grid":
        files, cols, tw, th, dst = v
        rows = (len(files) + cols - 1) // cols
        im = Image.new("RGB", (cols * tw, rows * th))
        for i, fn in enumerate(files):
            im.paste(Image.open(f(fn)).convert("RGB"), ((i % cols) * tw, (i // cols) * th))
        os.makedirs(os.path.dirname(f(dst)), exist_ok=True)
        im.save(f(dst))
    elif k == "jpg":
        src, dst, q = v
        Image.open(f(src)).convert("RGB").save(f(dst), "JPEG", quality=q, subsampling=2)
    elif k == "copy":
        shutil.copyfile(f(v[0]), f(v[1]))
    elif k == "crop":
        src, dst, x, y, w, h, s = v
        Image.open(f(src)).convert("RGB").crop((x, y, x + w, y + h)).resize((w * s, h * s), Image.NEAREST).save(f(dst))
    else:
        raise ValueError(k)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated names (with or without extension)")
    ap.add_argument("--out", default=VARS["DEFOUT"])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--keep-tmp", action="store_true")
    a = ap.parse_args()
    byname = {e["name"]: e for e in ENTRIES}
    if a.list:
        for e in ENTRIES:
            print(e["name"], "" if e.get("reproducible", True) else "NOT REPRODUCIBLE: " + e["reason"])
        return
    want = [e["name"] for e in ENTRIES]
    if a.only:
        sel = [s.strip() for s in a.only.split(",")]
        want = [n for n in byname if any(n == s or n.rsplit(".", 1)[0] == s for s in sel)]
    done, failed = set(), {}
    def run(n):
        if n in done: return
        e = byname[n]
        if not e.get("reproducible", True):
            print(f"[skip] {n}: {e['reason']}"); done.add(n); return
        for d in e.get("dep", []): run(d)
        T = os.path.join(a.out, "_tmp", n.replace("/", "_")); os.makedirs(T, exist_ok=True)
        t0 = time.time(); print(f"[{n}]", flush=True)
        try:
            for p in e["steps"]: do_step(p, T, a.out)
            print(f"  ok {time.time()-t0:.1f}s")
        except Exception as ex:
            failed[n] = str(ex); print(f"  ERROR: {ex}")
        done.add(n)
    for n in want: run(n)
    if not a.keep_tmp:
        shutil.rmtree(os.path.join(a.out, "_tmp"), ignore_errors=True)
    if failed:
        print("FAILURES:", *[f"{k}: {v}" for k, v in failed.items()], sep="\n  "); sys.exit(1)

if __name__ == "__main__":
    main()
