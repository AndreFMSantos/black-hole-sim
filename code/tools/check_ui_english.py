#!/usr/bin/env python3
"""ctest: the interactive UI (panel, HUD, window title) must be in English. Scans the string literals of ui.cpp and flight.hpp
(HUD) for Portuguese words and diacritics. Exit 1 on any hit.  Usage: check_ui_english.py <file>..."""
import re, sys
PT = set("""nao sem camara ceu raio tempo disco esta para com uma dos das buraco negro ajuda imagem distancia precisao
observador acabamento cinematografico exposicao tocar pasta instantaneo carga metrica verme versoes variantes sombra
ceu propulsao horizonte simulacao parada rato roda teclas esconde mostra""".split())
bad = 0
for path in sys.argv[1:]:
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        code = line.split("//")[0] if line.lstrip().startswith("//") else line
        for lit in re.findall(r'"((?:[^"\\]|\\.)*)"', code):
            words = set(re.findall(r"[a-zà-ú]+", lit.lower()))
            hit = words & PT
            if hit or re.search(r"[àáâãçéêíóôõú]", lit.lower()):
                print(f"{path}:{n}: Portuguese in UI string {lit[:60]!r} ({sorted(hit)})"); bad += 1
print("UI language check:", "FAILED" if bad else "OK (English)")
sys.exit(1 if bad else 0)
