import re
# Self-doubt or re-checking words in the monitor's own text.
HEDGE = re.compile(r"\b(?:wait|hmm+|however|double[- ]check|let me (?:verify|check|re-?check)|actually|did (?:we|they|I|it) miss|not sure|is (?:that|this) (?:right|correct)\?)", re.I)
def label(unit):
    return "yes" if HEDGE.search((unit.get("monitor_cot") or "") + "\n" + (unit.get("monitor_reply") or "")) else "no"
