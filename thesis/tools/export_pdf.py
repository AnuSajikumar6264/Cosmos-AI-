"""Export thesis/build/COSMOS_Thesis.docx to PDF with an up-to-date table of contents.

Run with LibreOffice's bundled Python (it provides the UNO bridge):

    /data/anupa/COSMOS_work/.tools_lo/opt/libreoffice26.8/program/python thesis/tools/export_pdf.py

LibreOffice is started headless. The script opens the document, updates every index (the table
of contents) and field so page numbers are filled in, and exports a PDF with heading bookmarks.
"""
import os
import subprocess
import sys
import time

import uno
from com.sun.star.beans import PropertyValue

HERE = os.path.dirname(os.path.abspath(__file__))
TH = os.path.dirname(HERE)
SOFFICE = "/data/anupa/COSMOS_work/.tools_lo/opt/libreoffice26.8/program/soffice"
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(TH, "build", "COSMOS_Thesis.docx")
DST = sys.argv[2] if len(sys.argv) > 2 else os.path.join(TH, "build", "COSMOS_Thesis.pdf")
PIPE = "cosmos_thesis_pdf"


def prop(name, value):
    p = PropertyValue()
    p.Name, p.Value = name, value
    return p


def main():
    proc = subprocess.Popen([SOFFICE, "--headless", "--norestore", "--nologo",
                             "-env:UserInstallation=file:///tmp/thesis_pdf/lo_profile_uno",
                             f"--accept=pipe,name={PIPE};urp;"])
    ctx = None
    local = uno.getComponentContext()
    resolver = local.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local)
    for _ in range(60):
        try:
            ctx = resolver.resolve(f"uno:pipe,name={PIPE};urp;StarOffice.ComponentContext")
            break
        except Exception:
            time.sleep(1)
    if ctx is None:
        proc.kill()
        raise SystemExit("could not connect to LibreOffice")
    desktop = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
    doc = desktop.loadComponentFromURL(uno.systemPathToFileUrl(os.path.abspath(SRC)), "_blank",
                                       0, (prop("Hidden", True),))
    # update fields and indexes twice: the TOC changes page breaks, which changes page numbers
    for _ in range(2):
        doc.getTextFields().refresh()
        idx = doc.getDocumentIndexes()
        for i in range(idx.getCount()):
            idx.getByIndex(i).update()
    print("indexes updated:", doc.getDocumentIndexes().getCount())
    filt = (prop("ExportBookmarks", True), prop("ExportNotes", False),
            prop("UseTaggedPDF", True), prop("SelectPdfVersion", 2),
            prop("EmbedStandardFonts", True))
    doc.storeToURL(uno.systemPathToFileUrl(os.path.abspath(DST)),
                   (prop("FilterName", "writer_pdf_Export"), prop("FilterData", uno.Any(
                       "[]com.sun.star.beans.PropertyValue", filt))))
    doc.close(True)
    try:
        desktop.terminate()
    except Exception:
        pass
    proc.wait(timeout=60)
    print("wrote", DST)


if __name__ == "__main__":
    main()
