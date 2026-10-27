"""Word reference document for the journal manuscript (journal_v2/tools/reference.docx).

Starts from the thesis reference document and changes it to a compact manuscript layout:
A4, 2.5 cm margins, 11 pt Times New Roman, 1.15 line spacing, 10 pt tables, 9.5 pt captions.

    python paper/journal_v2/tools/make_paper_reference.py
"""
import os
import shutil

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SRC = os.path.join(ROOT, "thesis", "tools", "reference.docx")
DST = os.path.join(HERE, "reference.docx")
FONT = "Times New Roman"

shutil.copy(SRC, DST)
d = Document(DST)
for s in d.sections:
    s.page_height, s.page_width = Cm(29.7), Cm(21.0)
    s.left_margin = s.right_margin = Cm(2.5)
    s.top_margin = s.bottom_margin = Cm(2.5)
st = d.styles
for name, size, spacing, after in (("Normal", 11, 1.15, 4), ("Body Text", 11, 1.15, 4),
                                   ("First Paragraph", 11, 1.15, 4), ("Compact", 11, 1.1, 1),
                                   ("Table Text", 9.5, 1.0, 0), ("Abstract", 10, 1.1, 4),
                                   ("Bibliography", 8.5, 1.0, 0.5)):
    if name in [x.name for x in st]:
        s = st[name]
        s.font.name, s.font.size = FONT, Pt(size)
        s.paragraph_format.line_spacing = spacing
        s.paragraph_format.space_after = Pt(after)
for name in ("Caption", "Image Caption", "Table Caption"):
    if name in [x.name for x in st]:
        s = st[name]
        s.font.name, s.font.size = FONT, Pt(9.5)
        s.paragraph_format.space_after = Pt(6)
        if name == "Table Caption":
            s.paragraph_format.keep_with_next = True
for name, size in (("Title", 16), ("Subtitle", 12), ("Heading 1", 12.5), ("Heading 2", 11.5),
                   ("Heading 3", 11)):
    if name in [x.name for x in st]:
        s = st[name]
        s.font.name, s.font.size, s.font.bold = FONT, Pt(size), True
        s.paragraph_format.space_before = Pt(10 if name.startswith("Heading 1") else 6)
        s.paragraph_format.space_after = Pt(4)
# notes placed below a table body (Elsevier: table footnotes below the table)
if "Table Note" not in [x.name for x in st]:
    tn = st.add_style("Table Note", WD_STYLE_TYPE.PARAGRAPH)
    tn.base_style = st["Normal"]
    tn.font.name, tn.font.size = FONT, Pt(9)
    tn.paragraph_format.space_before = Pt(2)
    tn.paragraph_format.space_after = Pt(10)
    tn.paragraph_format.line_spacing = 1.0
if "Author" in [x.name for x in st]:
    st["Author"].font.size = Pt(11)
    st["Author"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
d.save(DST)
print("wrote", DST)
