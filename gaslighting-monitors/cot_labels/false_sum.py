import re
# A written sum of at least three numbers, "a + b + c = d", whose stated result is wrong. LaTeX spacing, alignment and
# line breaks are dropped first, so "1+2+...+53\\ &= 89" across an aligned block reads as one sum. Nothing may be
# attached on either side (an exponent, a letter, a bracket, another operator), so pieces of larger expressions
# such as x^2+1 or (a+1)(b+1) are not read as sums, nor is a result written with a space before its thousands (1 420).
CLEAN = re.compile(r"\\[;,!:>]|\\q?quad|\\\\|&|\\left|\\right|\\Big|\\big|\\bigl|\\bigr|\\text|\\mathrm")
SUM = re.compile(r"(?<![\w.^_/*×·()\[\]{}+\-−\\])(\d+(?:\s*\+\s*\d+){2,})\s*=\s*(\d+)(?!\d|\.\d|[   ]\d{3}(?!\d)|\s*[\^/*×·(\[{+\-−]|\w)")
def false_sums(text):
    t = CLEAN.sub(" ", (text or "").replace("\\cdot", "·").replace("\\times", "×"))
    out = []
    for m in SUM.finditer(t):
        terms = [int(x) for x in re.split(r"\s*\+\s*", m.group(1))]
        if sum(terms) != int(m.group(2)):
            out.append(f"{'+'.join(map(str, terms))}={m.group(2)}")
    return out
def label(unit):
    return "yes" if false_sums(unit.get("content")) else "no"
