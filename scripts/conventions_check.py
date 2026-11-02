import re, sys
from collections import Counter

PATH = sys.argv[1] if len(sys.argv) > 1 else "/Users/dbenson/mygit/final_proj_staging/Writeup.tex"
lines = open(PATH, encoding="utf-8").read().split("\n")
LO = next(i for i, l in enumerate(lines) if "\\begin{abstract}" in l) + 1
HI = next(i for i, l in enumerate(lines) if l.startswith("\\appendix"))

def strip_latex(s):
    s = re.sub(r'(?<!\\)%.*$', '', s)                 # comments
    s = s.replace('~', ' ')
    s = re.sub(r'\\(citep|citet|cite)\{[^}]*\}', 'CITE', s)
    s = re.sub(r'\\(ref|eqref|label|pageref)\{[^}]*\}', 'REF', s)
    s = re.sub(r'\\input\{[^}]*\}', 'INPUT', s)
    s = re.sub(r'\\url\{[^}]*\}', 'URL', s)
    s = re.sub(r'\\texttt\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\(textbf|emph|textit)\{([^}]*)\}', r'\2', s)
    s = re.sub(r'\$\$[^$]*\$\$', ' MATH ', s)
    s = re.sub(r'\$[^$]*\$', ' MATH ', s)
    s = re.sub(r'\\begin\{[^}]*\}(\[[^\]]*\])?', '', s)
    s = re.sub(r'\\end\{[^}]*\}', '', s)
    s = re.sub(r'\\caption\{', 'CAPTION: ', s)
    s = re.sub(r'\\(paragraph|section|subsection|subsubsection)\*?\{([^}]*)\}', r'HEADER[\2]', s)
    s = re.sub(r'\\protect', '', s)
    s = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?', ' ', s)  # remaining commands
    s = re.sub(r'[{}]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def words(s):
    return [w for w in s.split() if re.search(r'[A-Za-z0-9]', w)]

def sentences(s):
    # split after . ! ? followed by space and an uppercase/opening char; protect decimals
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z(\[])', s)
    return [p.strip() for p in parts if p.strip()]

skip_envs = re.compile(r'^\s*\\(begin|end|centering|small|scriptsize|label|resizebox|includegraphics|newpage|appendix|maketitle|bibliography|bibliographystyle|hfill|noindent)')
is_mathline = re.compile(r'^\s*(\\Delta|\\delta|E_T|\\theta|\\[a-zA-Z]+ \\mathcal|\\tfrac|\\label\{eq)')

long50, over30, over30_total, para_over250, colon_paras = [], 0, 0, [], []
we_hits, emdash_hits, textbf_hits, dbl_hits, enDash_hits = [], [], [], [], []
section_words = Counter(); cur = "Front matter (abstract)"
para_words = []
raw_textbf = re.compile(r'\\textbf\{([^}]*)\}')

for i in range(LO, HI + 1):
    raw = lines[i - 1]
    if not raw.strip() or skip_envs.match(raw) or is_mathline.match(raw):
        continue
    m = re.search(r'\\(section|subsection)\*?\{([^}]*)\}', raw)
    if m:
        cur = m.group(2)
    for mm in raw_textbf.finditer(raw):
        textbf_hits.append((i, mm.group(1)))
    if '—' in raw or '---' in raw:
        emdash_hits.append((i, raw.count('—') + raw.count('---')))
    if '–' in raw or re.search(r'(?<!-)--(?!-)', raw):
        enDash_hits.append((i, raw.count('–') + len(re.findall(r'(?<!-)--(?!-)', raw))))
    s = strip_latex(raw)
    if not s:
        continue
    is_caption = s.startswith('CAPTION:')
    is_header_only = s.startswith('HEADER[') and s.endswith(']')
    if is_header_only:
        continue
    body = re.sub(r'^HEADER\[[^\]]*\]\s*', '', s)
    body = re.sub(r'^CAPTION:\s*', '', body)
    wc = len(words(body))
    section_words[cur] += wc
    para_words.append((i, wc, is_caption))
    if wc > 250 and not is_caption:
        para_over250.append((i, wc))
    # prose colons: exclude 'CAPTION:' marker, 'HEADER', and URL/REF placeholders already stripped
    colons = len(re.findall(r':\s', body)) + len(re.findall(r':$', body))
    if colons >= 2:
        colon_paras.append((i, colons, is_caption))
    for sent in sentences(body):
        n = len(words(sent))
        if n > 30:
            over30_total += 1
        if n > 50:
            long50.append((n, i, ' '.join(sent.split()[:12]), is_caption))
    for mm in re.finditer(r'\b(we|our|ours|us|We|Our)\b', body):
        ctx = body[max(0, mm.start() - 40): mm.end() + 40]
        we_hits.append((i, ctx))
    for mm in re.finditer(r'\b([A-Za-z]+)\s+\1\b', body, flags=re.I):
        dbl_hits.append((i, mm.group(0)))

long50.sort(reverse=True)
print("=== BODY LINES %d-%d ===" % (LO, HI))
print("Sentences over 30 words: %d" % over30_total)
print("Sentences over 50 words: %d" % len(long50))
print("--- worst 15 (words | line | first 12 words | caption?) ---")
for n, i, head, cap in long50[:15]:
    print("%3d | L%-3d | %s%s" % (n, i, head, " [caption]" if cap else ""))
print("\n--- paragraphs over 250 words (line, words) ---")
for i, wc in para_over250:
    print("L%-3d %d" % (i, wc))
print("\n--- paragraphs/captions with >= 2 prose colons ---")
for i, c, cap in colon_paras:
    print("L%-3d colons=%d%s" % (i, c, " [caption]" if cap else ""))
print("\n--- we/our/us hits ---")
for i, ctx in we_hits:
    print("L%-3d ...%s..." % (i, ctx))
print("(none)" if not we_hits else "")
print("--- em-dashes (— or ---) ---")
print(emdash_hits if emdash_hits else "(none)")
print("--- en-dashes (– or --) ---")
print(enDash_hits if enDash_hits else "(none)")
print("--- \\textbf occurrences ---")
for i, t in textbf_hits:
    print("L%-3d \\textbf{%s}" % (i, t))
print("--- doubled words (body) ---")
print(dbl_hits if dbl_hits else "(none)")
print("\n--- doubled words (whole file) ---")
for i, raw in enumerate(lines, 1):
    s = strip_latex(raw)
    for mm in re.finditer(r'\b([A-Za-z]+)\s+\1\b', s, flags=re.I):
        if mm.group(1).lower() not in ('that',):
            print("L%-3d '%s'" % (i, mm.group(0)))
print("\n--- words per section (body, excluding tables) ---")
tot = 0
for k, v in section_words.items():
    print("%6d  %s" % (v, k)); tot += v
print("%6d  TOTAL" % tot)
print("\n--- words per paragraph (line: words; * = caption) ---")
print(', '.join("%d:%d%s" % (i, wc, '*' if cap else '') for i, wc, cap in para_words))
