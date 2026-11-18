#pragma once
// v4.0: named presets. "interstellar" copies the documented choices of James, von Tunzelmann, Franklin & Thorne (2015, DNGR) for the film's black hole Gargantua:
//  spin a = 0.6, NO frequency shift (neither Doppler nor gravitational). The paper (and our --disk-temp uniform --smooth-disk) uses a flat 4500 K disk; the PRESET is the film LOOK instead (2026-11-18: a flat disk looked like a drawing): Novikov-Thorne falloff, filaments, soft rim.
// It is an ARTISTIC rendering, not physics: the paper itself says the film's disk is not Novikov-Thorne and the colour shift was switched off for the audience. The realistic counterpart
// (a = 0.9, Novikov-Thorne, everything physical) is the default of this app. Camera: the caller sets inclination and distance.
#include "render.hpp"

namespace bhs::kcuda {

inline void apply_preset_interstellar(Params& p) {
  p.a = 0.6; p.rin = -1.0; p.rout = 24.0;      // ISCO of a = 0.6 (3.83 M) outwards
  p.noshift = true; p.nodoppler = false;        // no Doppler and no gravitational redshift (the film's choice)
  // Film look (our approximation, not the paper's disk): a thin disk whose brightness and colour fall off with radius (Novikov-Thorne profile, gold near the ISCO,
  // orange-red further out) with fine azimuthal filaments, a soft irregular rim and no limb darkening; lensing is exact.
  p.nt = true; p.uniform_disk = false; p.tpeak = 4800.0; p.limb = false;
  p.smooth_disk = false; p.tex_amp = 0.55; p.tex_fr = 70.0; p.tex_fphi = 5.0; p.tex_oct = 5;
  p.edge_soft = 0.2; p.edge_noise = 0.15;
  p.hr = 0.0; p.model = 0; p.nodisk = false; p.charge = 0.0; p.lambda = 0.0; p.wormhole = false;
  p.exposure = 1.6;
}

}  // namespace bhs::kcuda
