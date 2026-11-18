"""List of the time-lapse segments (v1.1): chronological order, visible duration (s), tool, title and key number.

Each segment is rendered with the global camera C(t) (camera.py) over its interval [start, start + dur + OVERLAP]; the overlap of
OVERLAP seconds serves the cross-fade into the next segment, and since the camera only depends on the global time it is identical in both.
The key numbers come from the notes (notes/physics/*) or from measurements of this session; those marked {..} are filled in with the recorder's output.
"""
from dataclasses import dataclass, field

FPS = 30
OVERLAP = 1.0


@dataclass
class Seg:
    id: str
    tool: str            # rec2d | cpu | v05 | route
    dur: float           # visible seconds (without the overlap)
    title: str
    key: str             # key number (may have {braces} from the recorder's meta)
    args: dict = field(default_factory=dict)


SEGMENTS = [
    Seg("s01", "rec2d", 9, "v0.1 · Newton and integrators", "Euler spirals outward: r from 1.0 to {euler_r} at t = {t}; Verlet and RK4 stay at r = 1.000", {"scene": "newton", "steps": 32}),
    Seg("s02", "rec2d", 8, "v0.2 · Light bends around the black hole", "shadow: b = 5.19615 M (square root of 27) · {captured} of {n} rays captured", {"scene": "light"}),
    Seg("s03", "cpu", 9, "v0.3 · 3D ray tracer: the shadow over the stars", "1280×720 in 1.5 s (CPU, 16 threads)", {"app": "v0_3-raytrace-cpu", "w": 1280, "h": 720, "ssaa": 2}),
    Seg("s04", "cpu", 9, "v0.4 · Accretion disk", "ISCO at 6 M · blackbody colour, redshift and Doppler", {"app": "v0_4-accretion-disk", "w": 1280, "h": 720, "ssaa": 2}),
    Seg("s05", "v05", 8, "v0.5 · Real time on the GPU", "1361 fps at 1080p (RTX 5080)", {"sky": False}),
    Seg("s06", "v05", 7, "v0.5b · NASA's real sky", "8192×4096 HDR map, 14 mipmap levels, quad filtering", {"sky": True}),
    Seg("s07", "route", 12, "v0.6 · Kerr black hole (a = 0.9)", "Carter geodesics in CUDA: 7 ms per frame at 1080p (float)", {"flags": ["--profile", "newtonian"], "samples": 16}),
    Seg("s08", "route", 12, "v0.7 · Novikov–Thorne and photon rings", "luminosity = 1 − E_isco to 1e-5 · ring ratio = e^−π = 0.04321", {"flags": ["--profile", "nt"], "samples": 16}),
    Seg("s09", "route", 8, "v0.7d/v0.8d · Cinematic finish", "64 samples in 0.5 s (178 s on the CPU)", {"flags": ["--cinematic"], "samples": 16}),
    Seg("s10", "route", 9, "v0.9b · Disk with thickness", "radiative transfer along the ray: from 29.6 to 4.4 ms per sample (FP64 to float)",
        {"flags": ["--cinematic"], "samples": 16, "disk": {"hr": 0.06, "tau": 4, "flare": 0.15}}),
    Seg("s11", "route", 12, "v0.9c · Turbulence, hot spot and slow light", "the disk rotates with Kerr's Ω(r); soft, irregular edge",
        {"flags": ["--cinematic"], "samples": 16, "disk": {"hr": 0.06, "tau": 4, "flare": 0.15, "turb": 1.2, "turb_scale": 1.2, "edge_soft": 0.12,
                                                         "spot": {"r": 9, "phi": 200, "size": 1.2, "amp": 5}, "slowlight": True}}),
    Seg("s11b", "radio", 12, "v2.2b · Radio mode: a Fishbone–Moncrief torus", "thermal synchrotron (Dexter 2016) · brightness temperature at 230 GHz, no sky",
        {"kind": "torus"}),
    Seg("s11c", "radio", 20, "v2.2g · Our own GRMHD (AthenaK, MAD a = 0.9375)", "600 M with a snapshot every 2 M · ring of 37–38 µas ~ Kerr critical curve (37.4 µas)",
        {"kind": "grmhd_hi", "n": 300}),
    Seg("s11f", "var", 14, "v2.1d · Wormhole (Interstellar metric)", "ρ = 1, 2a = 0.01ρ, W = 0.05ρ · GPU vs CPU: max error in ψ 5e-5 · two skies: NASA and WISE", {"kind": "wh"}),
    Seg("s11g", "var", 12, "v2.1a · Electric charge: Reissner–Nordström", "shadow from 5.196 M (Q = 0) to 4.180 M (Q² = 0.9) · no horizon (Q > 1): no shadow", {"kind": "rn"}),
    Seg("s11h", "var", 8, "v2.1c · Cosmological constant: Schwarzschild–de Sitter", "shadow: b = sqrt(27) M / sqrt(1 − 9 Λ M²) · GPU vs analytic to 5e-7", {"kind": "sds"}),
]



def starts():
    t = 0.0
    out = {}
    for s in SEGMENTS:
        out[s.id] = t
        t += s.dur
    return out, t   # start of each segment and total duration
