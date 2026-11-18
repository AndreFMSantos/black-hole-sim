#include <bhs/turbulence.hpp>
#include <cstdio>
int main(){ double s=0,s2=0; const int n=2000000; uint32_t h=123u; auto rnd=[&](){h=h*1664525u+1013904223u; return double(h>>8)/16777216.0;};
 for(int i=0;i<n;++i){ double v=bhs::turb::fbm(60*rnd(),60*rnd(),60*rnd()); s+=v; s2+=v*v;} std::printf("mean %.5f var %.5f\n", s/n, s2/n-(s/n)*(s/n)); }
