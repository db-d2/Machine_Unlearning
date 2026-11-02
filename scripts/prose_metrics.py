import re, sys
path = sys.argv[1] if len(sys.argv) > 1 else 'Writeup.tex'
src = open(path).read().split('\n')
body_start = next(i for i, l in enumerate(src) if '\\begin{abstract}' in l) + 1
body_end = next(i for i, l in enumerate(src) if l.startswith('\\appendix'))   # abstract .. before \appendix
def strip(line):
    s = line
    s = re.sub(r'%.*$', '', s)
    s = re.sub(r'\\(ref|eqref|label|cite[tp]?|input|includegraphics|protect|resizebox|caption)\*?(\[[^\]]*\])?\{[^}]*\}', ' REF ', s)
    s = re.sub(r'\$[^$]*\$', ' MATH ', s)
    s = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', s)
    s = re.sub(r'[{}]', '', s)
    return s
paras = []
for i, line in enumerate(src, 1):
    if not (body_start <= i <= body_end): continue
    if re.match(r'\s*\\(begin|end|section|subsection|input|resizebox|label|centering|appendix)', line): continue
    t = strip(line).strip()
    if len(t.split()) < 25: continue
    paras.append((i, t, line.count(':') - len(re.findall(r'\\(ref|eqref|label|cite[tp]?)\{[^}]*\}', line)) - len(re.findall(r'\\(ref|eqref|label)\{[^}]*:', line))*0))
# recount colons properly: colons in stripped text
sent_lens = []; long30 = long50 = 0; paras_over_250 = []; colon_rows = []
for i, t, _ in paras:
    sents = [s for s in re.split(r'(?<=[.!?])\s+(?=[A-Z(])', t) if s.strip()]
    n_words = len(t.split())
    if n_words > 250: paras_over_250.append((i, n_words))
    cols = t.count(':')
    if cols >= 2: colon_rows.append((i, cols, n_words))
    for s in sents:
        w = len(s.split()); sent_lens.append((w, i, s[:90]))
        if w > 30: long30 += 1
        if w > 50: long50 += 1
sent_lens.sort(reverse=True)
import statistics
print(f"body paragraphs: {len(paras)}; sentences: {len(sent_lens)}; median words {statistics.median([w for w,_,_ in sent_lens])}")
print(f"sentences >30 words: {long30}; >50 words: {long50}")
print("longest sentences:")
for w,i,s in sent_lens[:12]: print(f"  {w:3d} words  line {i}: {s}...")
print(f"paragraphs >250 words: {len(paras_over_250)} -> {paras_over_250}")
print(f"paragraphs with >=2 prose colons: {len(colon_rows)}")
for i,c,n in colon_rows: print(f"  line {i}: {c} colons in {n} words")
