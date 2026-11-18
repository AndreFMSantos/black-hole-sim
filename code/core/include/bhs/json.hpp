#pragma once
// Minimal JSON (RFC 8259) for the route files: objects, arrays, numbers, strings (with \uXXXX in the BMP), true/false/null.
// No dependencies; errors with a position. Neither fast nor complete (no surrogates), and enough for small files.
#include <cstdlib>
#include <string>
#include <vector>

namespace bhs::json {

struct Value {
  enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
  bool b = false;
  double n = 0;
  std::string s;
  std::vector<Value> a;
  std::vector<std::string> keys;  // objects: keys and values in parallel vectors (insertion order)
  std::vector<Value> vals;

  const Value* get(const std::string& k) const {
    for (size_t i = 0; i < keys.size(); ++i) if (keys[i] == k) return &vals[i];
    return nullptr;
  }
  double num(const std::string& k, double def) const { const Value* v = get(k); return v && v->type == Num ? v->n : def; }
  std::string str(const std::string& k, const std::string& def) const { const Value* v = get(k); return v && v->type == Str ? v->s : def; }
  bool boolean(const std::string& k, bool def) const { const Value* v = get(k); return v && v->type == Bool ? v->b : def; }
};

namespace detail {
struct Parser {
  const std::string& t;
  size_t i = 0;
  std::string err;
  explicit Parser(const std::string& s) : t(s) {}
  bool fail(const std::string& m) { if (err.empty()) err = m + " (position " + std::to_string(i) + ")"; return false; }
  void ws() { while (i < t.size() && (t[i] == ' ' || t[i] == '\t' || t[i] == '\n' || t[i] == '\r')) ++i; }
  bool lit(const char* w) { size_t n = 0; while (w[n]) ++n; if (t.compare(i, n, w) == 0) { i += n; return true; } return fail(std::string("expected '") + w + "'"); }

  bool str(std::string& out) {
    if (i >= t.size() || t[i] != '"') return fail("expected a string");
    ++i; out.clear();
    while (i < t.size()) {
      const char c = t[i++];
      if (c == '"') return true;
      if (c == '\\') {
        if (i >= t.size()) return fail("incomplete escape");
        const char e = t[i++];
        switch (e) {
          case '"': out += '"'; break; case '\\': out += '\\'; break; case '/': out += '/'; break;
          case 'b': out += '\b'; break; case 'f': out += '\f'; break; case 'n': out += '\n'; break;
          case 'r': out += '\r'; break; case 't': out += '\t'; break;
          case 'u': {
            if (i + 4 > t.size()) return fail("incomplete \\u");
            unsigned cp = 0;
            for (int k = 0; k < 4; ++k) {
              const char h = t[i++]; cp <<= 4;
              if (h >= '0' && h <= '9') cp |= unsigned(h - '0'); else if (h >= 'a' && h <= 'f') cp |= unsigned(h - 'a' + 10);
              else if (h >= 'A' && h <= 'F') cp |= unsigned(h - 'A' + 10); else return fail("invalid hex in \\u");
            }
            if (cp < 0x80) out += char(cp);
            else if (cp < 0x800) { out += char(0xC0 | (cp >> 6)); out += char(0x80 | (cp & 0x3F)); }
            else { out += char(0xE0 | (cp >> 12)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
            break;
          }
          default: return fail("invalid escape");
        }
      } else out += c;
    }
    return fail("unterminated string");
  }

  bool number(double& d) {
    const size_t s = i;
    if (i < t.size() && t[i] == '-') ++i;
    if (i >= t.size()) return fail("incomplete number");
    if (t[i] == '0') ++i;
    else if (t[i] >= '1' && t[i] <= '9') { while (i < t.size() && t[i] >= '0' && t[i] <= '9') ++i; }
    else return fail("invalid number");
    if (i < t.size() && t[i] == '.') { ++i; const size_t f = i; while (i < t.size() && t[i] >= '0' && t[i] <= '9') ++i; if (i == f) return fail("fraction without digits"); }
    if (i < t.size() && (t[i] == 'e' || t[i] == 'E')) {
      ++i; if (i < t.size() && (t[i] == '+' || t[i] == '-')) ++i;
      const size_t f = i; while (i < t.size() && t[i] >= '0' && t[i] <= '9') ++i; if (i == f) return fail("exponent without digits");
    }
    d = std::strtod(t.substr(s, i - s).c_str(), nullptr);
    return true;
  }

  bool value(Value& v, int depth) {
    if (depth > 64) return fail("nesting too deep");
    ws();
    if (i >= t.size()) return fail("unexpected end");
    const char c = t[i];
    if (c == '{') {
      v.type = Value::Obj; ++i; ws();
      if (i < t.size() && t[i] == '}') { ++i; return true; }
      for (;;) {
        ws(); std::string k; if (!str(k)) return false;
        ws(); if (i >= t.size() || t[i] != ':') return fail("expected ':'");
        ++i; Value x; if (!value(x, depth + 1)) return false;
        v.keys.push_back(k); v.vals.push_back(std::move(x));
        ws(); if (i < t.size() && t[i] == ',') { ++i; continue; }
        if (i < t.size() && t[i] == '}') { ++i; return true; }
        return fail("expected ',' or '}'");
      }
    }
    if (c == '[') {
      v.type = Value::Arr; ++i; ws();
      if (i < t.size() && t[i] == ']') { ++i; return true; }
      for (;;) {
        Value x; if (!value(x, depth + 1)) return false;
        v.a.push_back(std::move(x));
        ws(); if (i < t.size() && t[i] == ',') { ++i; continue; }
        if (i < t.size() && t[i] == ']') { ++i; return true; }
        return fail("expected ',' or ']'");
      }
    }
    if (c == '"') { v.type = Value::Str; return str(v.s); }
    if (c == 't') { v.type = Value::Bool; v.b = true; return lit("true"); }
    if (c == 'f') { v.type = Value::Bool; v.b = false; return lit("false"); }
    if (c == 'n') { v.type = Value::Null; return lit("null"); }
    v.type = Value::Num; return number(v.n);
  }
};
}  // namespace detail

inline bool parse(const std::string& text, Value& out, std::string& err) {
  detail::Parser p(text);
  out = Value{};
  if (!p.value(out, 0)) { err = p.err; return false; }
  p.ws();
  if (p.i != text.size()) { p.fail("extra text after the value"); err = p.err; return false; }
  return true;
}

}  // namespace bhs::json
