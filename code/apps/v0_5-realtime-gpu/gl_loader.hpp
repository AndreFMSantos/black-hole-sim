#pragma once
// Minimal OpenGL 4.6 core loader: only the ~35 functions used, via glfwGetProcAddress, with the
// PFN*PROC types of the official <GL/glcorearb.h> header (Khronos, already installed with Mesa).
// No glad/glew: nothing to vendor or generate. Adding a function = one line in BHS_GL_FUNCS.
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GL/glcorearb.h>

#include <cstdio>

#define BHS_GL_FUNCS(X) \
  X(GetString, GETSTRING) X(Finish, FINISH) X(Viewport, VIEWPORT) \
  X(CreateShader, CREATESHADER) X(ShaderSource, SHADERSOURCE) X(CompileShader, COMPILESHADER) \
  X(GetShaderiv, GETSHADERIV) X(GetShaderInfoLog, GETSHADERINFOLOG) X(DeleteShader, DELETESHADER) \
  X(CreateProgram, CREATEPROGRAM) X(AttachShader, ATTACHSHADER) X(LinkProgram, LINKPROGRAM) \
  X(GetProgramiv, GETPROGRAMIV) X(GetProgramInfoLog, GETPROGRAMINFOLOG) X(UseProgram, USEPROGRAM) \
  X(GetUniformLocation, GETUNIFORMLOCATION) \
  X(ProgramUniform1f, PROGRAMUNIFORM1F) X(ProgramUniform1i, PROGRAMUNIFORM1I) \
  X(ProgramUniform2i, PROGRAMUNIFORM2I) X(ProgramUniform3f, PROGRAMUNIFORM3F) \
  X(DispatchCompute, DISPATCHCOMPUTE) X(MemoryBarrier, MEMORYBARRIER) \
  X(CreateTextures, CREATETEXTURES) X(DeleteTextures, DELETETEXTURES) \
  X(TextureStorage1D, TEXTURESTORAGE1D) X(TextureStorage2D, TEXTURESTORAGE2D) \
  X(TextureSubImage1D, TEXTURESUBIMAGE1D) X(TextureParameteri, TEXTUREPARAMETERI) \
  X(BindTextureUnit, BINDTEXTUREUNIT) X(BindImageTexture, BINDIMAGETEXTURE) \
  X(GetTextureImage, GETTEXTUREIMAGE) \
  X(CreateFramebuffers, CREATEFRAMEBUFFERS) X(NamedFramebufferTexture, NAMEDFRAMEBUFFERTEXTURE) \
  X(BlitNamedFramebuffer, BLITNAMEDFRAMEBUFFER) X(CreateVertexArrays, CREATEVERTEXARRAYS) \
  X(BindVertexArray, BINDVERTEXARRAY) \
  X(TextureSubImage2D, TEXTURESUBIMAGE2D) X(GenerateTextureMipmap, GENERATETEXTUREMIPMAP) \
  X(TextureParameterf, TEXTUREPARAMETERF) X(GetFloatv, GETFLOATV)

namespace gl {
#define X(name, upper) inline PFNGL##upper##PROC name = nullptr;
BHS_GL_FUNCS(X)
#undef X

inline bool load() {
  bool ok = true;
#define X(name, upper) \
  name = reinterpret_cast<PFNGL##upper##PROC>(glfwGetProcAddress("gl" #name)); \
  if (!name) { std::fprintf(stderr, "gl%s not found\n", #name); ok = false; }
  BHS_GL_FUNCS(X)
#undef X
  return ok;
}
}  // namespace gl
