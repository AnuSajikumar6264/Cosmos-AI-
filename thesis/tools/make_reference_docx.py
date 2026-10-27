"""Create thesis/tools/reference.docx: the Word style template pandoc uses for the thesis.

A4, 3.0 cm binding margin, 2.5 cm elsewhere; Times New Roman 12 pt body at 1.5 line spacing,
justified; numbered headings in the same family; captions 10 pt italic.
"""
import os
import subprocess

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
PANDOC = "/data/anupa/COSMOS_work/.tools/pandoc311/bin/pandoc"
out = os.path.join(HERE, "reference.docx")
with open(out, "wb") as f:
    f.write(subprocess.run([PANDOC, "--print-default-data-file", "reference.docx"],
                           check=True, capture_output=True).stdout)
doc = Document(out)
for s in doc.sections:
    s.page_height, s.page_width = Cm(29.7), Cm(21.0)
    s.left_margin, s.right_margin = Cm(3.0), Cm(2.5)
    s.top_margin, s.bottom_margin = Cm(2.5), Cm(2.5)
FONT = "Times New Roman"
st = doc.styles
for name in ("Normal", "Body Text", "First Paragraph", "Block Text"):
    if name in [x.name for x in st]:
        s = st[name]
        s.font.name, s.font.size = FONT, Pt(12)
        pf = s.paragraph_format
        pf.line_spacing = 1.5
        pf.space_after = Pt(6)
        if name in ("Body Text", "First Paragraph"):
            pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
if "Compact" in [x.name for x in st]:                     # tight lists (tables are restyled later)
    c = st["Compact"]
    c.font.name, c.font.size = FONT, Pt(12)
    c.paragraph_format.line_spacing = 1.3
    c.paragraph_format.space_after = Pt(2)
    c.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
from docx.enum.style import WD_STYLE_TYPE
if "Table Text" not in [x.name for x in st]:
    t = st.add_style("Table Text", WD_STYLE_TYPE.PARAGRAPH)
    t.base_style = st["Normal"]
    t.font.name, t.font.size = FONT, Pt(10)
    t.paragraph_format.line_spacing = 1.0
    t.paragraph_format.space_after = Pt(1)
    t.paragraph_format.space_before = Pt(1)
    t.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
if "List Entry" not in [x.name for x in st]:
    le = st.add_style("List Entry", WD_STYLE_TYPE.PARAGRAPH)   # entries of the lists of figures/tables
    le.base_style = st["Normal"]
    le.font.name, le.font.size = FONT, Pt(12)
    le.paragraph_format.line_spacing = 1.15
    le.paragraph_format.space_after = Pt(4)
    le.paragraph_format.left_indent = Cm(1.8)
    le.paragraph_format.first_line_indent = Cm(-1.8)
    le.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
for name, size in (("Title", 20), ("Subtitle", 14), ("Heading 1", 16), ("Heading 2", 14),
                   ("Heading 3", 12), ("Heading 4", 12)):
    if name in [x.name for x in st]:
        s = st[name]
        s.font.name, s.font.size, s.font.bold = FONT, Pt(size), True
        s.font.color.rgb = RGBColor(0, 0, 0)
        s.paragraph_format.space_before = Pt(18 if name == "Heading 1" else 12)
        s.paragraph_format.space_after = Pt(6)
        s.paragraph_format.keep_with_next = True
for name in ("Caption", "Image Caption", "Table Caption"):
    if name in [x.name for x in st]:
        s = st[name]
        s.font.name, s.font.size, s.font.italic = FONT, Pt(10), True
        s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        s.paragraph_format.line_spacing = 1.0
for name in ("Source Code", "Verbatim Char"):
    if name in [x.name for x in st]:
        st[name].font.size = Pt(9)
if "Bibliography" in [x.name for x in st]:
    st["Bibliography"].font.size = Pt(11)
    st["Bibliography"].paragraph_format.line_spacing = 1.15
from docx.oxml.ns import qn
for s in doc.styles:
    try:
        rpr = s.element.get_or_add_rPr()
    except Exception:
        continue
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        continue
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        rf.attrib.pop(qn(a), None)
    if s.name.startswith(("Heading", "Title", "Subtitle", "TOC")):
        rf.set(qn("w:ascii"), FONT)
        rf.set(qn("w:hAnsi"), FONT)
doc.save(out)
print("wrote", out)
