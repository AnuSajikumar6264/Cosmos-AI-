"""Render the journal manuscript: evidence placeholders -> Word -> (separately) PDF.

    python paper/journal_v2/tools/render_paper.py
    /data/anupa/COSMOS_work/.tools_lo/opt/libreoffice26.8/program/python thesis/tools/export_pdf.py \
        paper/journal_v2/build/manuscript.docx paper/journal_v2/build/manuscript.pdf

Numbers in src/paper.md are {{evidence.key[:format]}} placeholders resolved from
thesis/evidence/evidence.json, with the same rules as the thesis (thesis/tools/render.py):
a placeholder without evidence aborts the build. Figures are the separate artwork files in
artwork/ (make_paper_figures.py); supplementary tables are cited as [S:label], resolved from
src/supplementary_labels.json (make_supplementary.py).
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PAPER))
TH = os.path.join(ROOT, "thesis")
sys.path.insert(0, os.path.join(TH, "tools"))
import render as thesis_render  # noqa: E402  (substitute, fmt, fit_columns)

XREF_META = os.path.join(HERE, "crossref.yaml")
SUPP = re.compile(r"\[S:([A-Za-z0-9_\-]+)\]")


def check_front(body):
    """Elsevier limits: abstract at most 250 words, 1-6 keywords without 'and'/'of'."""
    m = re.search(r"abstract: \|\n(.*?)\n\s*\*\*Keywords:\*\*(.*?)\n", body, re.S)
    if not m:
        raise SystemExit("abstract or keywords not found")
    words = len(re.sub(r"\$[^$]*\$", "F", m.group(1)).split())
    kws = [k.strip() for k in m.group(2).split(";") if k.strip()]
    print(f"abstract: {words} words; keywords: {len(kws)}")
    if words > 250 or not 1 <= len(kws) <= 6 or any(re.search(r"\b(and|of)\b", k) for k in kws):
        raise SystemExit("abstract or keywords outside the journal limits")


def restyle(path, appendix_eqs=None):
    """Table style and widths; equation tables (pandoc-crossref tableEqns) get a narrow number
    column, and appendix equations are relabelled (A.1), (A.2), ... as the journal requires."""
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm
    d = Document(path)
    for tb in d.tables:
        cells = [c for row in tb.rows for c in row.cells]
        if len(tb.columns) == 2 and len(tb.rows) == 1 and all(c._tc.findall(".//" + qn("m:oMath")) for c in cells):
            for c, wcm in zip(cells, (14.6, 1.4)):
                c.width = Cm(wcm)
            for col, wcm in zip(tb.columns, (14.6, 1.4)):
                col.width = Cm(wcm)
            num = cells[1]._tc.findall(".//" + qn("m:t"))
            text = "".join(t.text or "" for t in num).strip()
            if appendix_eqs and text.strip("()") in appendix_eqs:
                num[0].text = "(" + appendix_eqs[text.strip("()")] + ")"
                for t in num[1:]:
                    t.text = ""
            continue
        for row in tb.rows:
            pr = row._tr.get_or_add_trPr()
            if pr.find(qn("w:cantSplit")) is None:
                pr.append(OxmlElement("w:cantSplit"))
            last = row._tr is tb.rows[-1]._tr
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.style = d.styles["Table Text"]
                    p.paragraph_format.keep_with_next = not last   # keep each table on one page
        thesis_render.fit_columns(tb, total_cm=16.0, cm_per_char=0.18)
        widen_narrow(tb)
    d.save(path)


def widen_narrow(tb):
    """Give every column room for its longest header word ("Arm", "ADTQC"), taken from the widest."""
    from docx.oxml.ns import qn
    from docx.shared import Cm
    cols = tb._tbl.tblGrid.findall(qn("w:gridCol"))
    w = [int(c.get(qn("w:w"))) / 1440 * 2.54 for c in cols]
    if len(w) < 2:
        return
    head = [c.text.split() for c in tb.rows[0].cells[:len(w)]]
    def word_cm(x):                                # rough widths of bold 9.5 pt Times letters
        return sum(0.25 if ch in "mwMW" else 0.22 if ch.isupper() else 0.15 if ch.isdigit() else 0.14
                   for ch in x)
    need_w = [max([word_cm(x) for x in h] + [0.4]) + 0.5 for h in head]
    for row in tb.rows[1:]:                        # short labels such as "M1 A1" stay on one line
        for j, c in enumerate(row.cells[:len(w)]):
            t = c.text.strip()
            if len(t) <= 18:
                need_w[j] = max(need_w[j], word_cm(t) * 1.05 + 0.6)
    short = [j for j in range(len(w)) if w[j] < need_w[j]]
    need = sum(need_w[j] - w[j] for j in short)
    spare = {j: w[j] - need_w[j] for j in range(len(w)) if w[j] > need_w[j]}
    if not short or sum(spare.values()) < need:
        return
    tot = sum(spare.values())
    for j in short:
        w[j] = need_w[j]
    for j, s in spare.items():                     # take the deficit from columns with room, pro rata
        w[j] -= need * s / tot
    for j, c in enumerate(cols):
        c.set(qn("w:w"), str(int(w[j] / 2.54 * 1440)))
    for row in tb.rows:
        for j, cell in enumerate(row.cells[:len(w)]):
            cell.width = Cm(w[j])


def main():
    ev = json.load(open(os.path.join(TH, "evidence", "evidence.json")))
    os.makedirs(os.path.join(PAPER, "build"), exist_ok=True)
    src = open(os.path.join(PAPER, "src", "paper.md"), encoding="utf-8").read()
    draft = "--draft" in sys.argv          # layout check only: pending text and numbers shown as [pending]
    if draft:
        src = re.sub(r"\[\[([A-Z0-9_]+)\]\]", r"[pending \1]", src)
        src = thesis_render.PH.sub(lambda m: m.group(0) if m.group(1) in ev and ev[m.group(1)]["value"] is not None
                                   else "[pending]", src)
    if "[[" in src:
        raise SystemExit("unfilled [[...]] markers: " + ", ".join(sorted(set(re.findall(r"\[\[[A-Z0-9_]+\]\]", src)))))
    labels = json.load(open(os.path.join(PAPER, "src", "supplementary_labels.json")))
    bad = [x for x in SUPP.findall(src) if x not in labels]
    if bad:
        raise SystemExit(f"unknown supplementary labels: {bad}")
    src = SUPP.sub(lambda m: "Supplementary " + labels[m.group(1)], src)
    missing = [m.group(1) for m in thesis_render.PH.finditer(src)
               if m.group(1) not in ev or ev[m.group(1)]["value"] is None]
    if missing:
        for k in missing:
            print("MISSING evidence:", k)
        raise SystemExit(f"{len(missing)} placeholders without evidence; build aborted")
    used = {}
    body = thesis_render.substitute(src, ev, used)
    check_front(body)
    md = os.path.join(PAPER, "build", "manuscript.md")
    open(md, "w", encoding="utf-8").write(body)
    json.dump(used, open(os.path.join(PAPER, "build", "used_keys.json"), "w"), indent=1)
    words = len(re.sub(r"(?s)^---.*?---", "", body.split("# References")[0]).split())
    # journal bibliography and author-year style when available (refs_journal.bib, elsevier-harvard.csl)
    bib = os.path.join(PAPER, "refs_journal.bib")
    bib = bib if os.path.isfile(bib) else os.path.join(TH, "refs.bib")
    csl = os.path.join(HERE, "elsevier-harvard.csl")
    csl = csl if os.path.isfile(csl) else os.path.join(TH, "tools", "ieee.csl")
    print("bibliography:", os.path.relpath(bib, ROOT), "style:", os.path.basename(csl))
    out = os.path.join(PAPER, "build", "manuscript.docx")
    cmd = [thesis_render.PANDOC, md, "--from", "markdown+tex_math_dollars+pipe_tables",
           "--filter", thesis_render.XREF, "--citeproc",
           "--bibliography", bib, "--csl", csl,
           "--reference-doc", os.path.join(HERE, "reference.docx"),
           "--resource-path", PAPER, "--number-sections", "--columns=10000",
           "--metadata-file", XREF_META, "-o", out]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=PAPER)
    if r.returncode:
        print(r.stderr)
        raise SystemExit("pandoc failed")
    for line in [x for x in r.stderr.splitlines() if x.strip()][:30]:
        print("pandoc:", line)
    # appendix equations carry ids eq:A1, eq:B1, ...: map their running numbers to A.1, B.1, ...
    eq_ids = re.findall(r"\{#eq:([A-Za-z0-9_\-]+)\}", body)
    appendix = {str(i + 1): f"{e[0]}.{e[1:]}" for i, e in enumerate(eq_ids) if re.fullmatch(r"[A-Z]\d+", e)}
    restyle(out, appendix)
    # Elsevier-style highlights (separate file; each bullet at most 85 characters)
    hl = thesis_render.substitute(open(os.path.join(PAPER, "src", "highlights.md"), encoding="utf-8").read(), ev, used)
    open(os.path.join(PAPER, "build", "highlights.md"), "w", encoding="utf-8").write(hl)
    for line in hl.splitlines():
        n = len(line.lstrip("- ").replace("$F_{0.5}$", "F0.5"))
        if n > 85:
            print(f"highlight too long ({n} chars): {line}")
    subprocess.run([thesis_render.PANDOC, os.path.join(PAPER, "build", "highlights.md"), "-o",
                    os.path.join(PAPER, "build", "highlights.docx"),
                    "--reference-doc", os.path.join(HERE, "reference.docx")], check=True)
    # supplementary material, from the same evidence file
    sup_src = os.path.join(PAPER, "src", "supplementary.md")
    if os.path.isfile(sup_src):
        sup = thesis_render.substitute(open(sup_src, encoding="utf-8").read(), ev, used)
        smd = os.path.join(PAPER, "build", "Supplementary_Material.md")
        open(smd, "w", encoding="utf-8").write(sup)
        r2 = subprocess.run([thesis_render.PANDOC, smd, "--from", "markdown+tex_math_dollars+pipe_tables",
                             "--filter", thesis_render.XREF, "--citeproc", "--bibliography", bib, "--csl", csl,
                             "--reference-doc", os.path.join(HERE, "reference.docx"), "--columns=10000",
                             "--resource-path", PAPER, "-o", os.path.join(PAPER, "build", "Supplementary_Material.docx")],
                            capture_output=True, text=True, cwd=PAPER)
        if r2.returncode:
            print(r2.stderr)
            raise SystemExit("pandoc failed on the supplementary material")
        restyle(os.path.join(PAPER, "build", "Supplementary_Material.docx"))
    print(f"{len(used)} evidence values used; about {words} words before the references; wrote {out}")


if __name__ == "__main__":
    main()
