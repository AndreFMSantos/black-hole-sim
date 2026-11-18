#pragma once
// Minimal OpenGL 4.1 Core loader for macOS (no GL 4.6 DSA).
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <OpenGL/gl3.h>

namespace gl {
inline bool load() { return glGetString(GL_VERSION) != nullptr; }
inline const GLubyte* GetString(GLenum name) { return glGetString(name); }
inline void GenFramebuffers(GLsizei n, GLuint* fbos) { glGenFramebuffers(n, fbos); }
inline void BindFramebuffer(GLenum target, GLuint fbo) { glBindFramebuffer(target, fbo); }
inline void GenTextures(GLsizei n, GLuint* texs) { glGenTextures(n, texs); }
inline void DeleteTextures(GLsizei n, const GLuint* texs) { glDeleteTextures(n, texs); }
inline void BindTexture(GLenum target, GLuint tex) { glBindTexture(target, tex); }
inline void TexImage2D(GLenum target, GLint level, GLint internalformat, GLsizei w, GLsizei h, GLint border, GLenum format, GLenum type, const void* data) {
  glTexImage2D(target, level, internalformat, w, h, border, format, type, data);
}
inline void TexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei w, GLsizei h, GLenum format, GLenum type, const void* data) {
  glTexSubImage2D(target, level, xoffset, yoffset, w, h, format, type, data);
}
inline void TexParameteri(GLenum target, GLenum pname, GLint param) { glTexParameteri(target, pname, param); }
inline void FramebufferTexture2D(GLenum target, GLenum attachment, GLenum textarget, GLuint tex, GLint level) {
  glFramebufferTexture2D(target, attachment, textarget, tex, level);
}
inline void BlitFramebuffer(GLint sx0, GLint sy0, GLint sx1, GLint sy1, GLint dx0, GLint dy0, GLint dx1, GLint dy1, GLbitfield mask, GLenum filter) {
  glBlitFramebuffer(sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, mask, filter);
}
}  // namespace gl
