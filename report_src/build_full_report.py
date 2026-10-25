"""Assemble the final report:

  Title -> Contents (+ Chapters 4-5) -> Ch1 -> Ch2 -> Ch3
  -> Ch4 Experiments and Results -> Ch5 Conclusion -> Bibliography

report.pdf (Chapters 1-3, LaTeX-built) is edited in place where its text no
longer matches the system as built. Old text is truly removed with PyMuPDF
redactions (so it is gone from copy/search too, not only hidden), and the new
text is typeset on the original baselines, in the same Latin Modern font and
size, keeping the same number of lines so nothing else on the page moves.
Chapter 4, Chapter 5 and the Bibliography come from chapter4_implementation.pdf
(build_pdf.py); its first page is laid over the free part of the last
original page, right below Section 3.3.

Run from report_src/:  python build_pdf.py && python build_full_report.py
"""

import io
import os
import re

import pymupdf
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "..", "report.pdf")
CH4 = os.path.join(HERE, "chapter4_implementation.pdf")
OUT = os.path.join(HERE, "..", "report_full.pdf")

FONTS = os.path.join(HERE, "fonts")
pdfmetrics.registerFont(TTFont("LM", os.path.join(FONTS, "lmroman10-regular.ttf")))
pdfmetrics.registerFont(TTFont("LM-Bold", os.path.join(FONTS, "lmroman10-bold.ttf")))

PAGE_W, PAGE_H = letter
BODY = 10.95   # body text size of report.pdf
TABLE = 9.96   # table text size
CH4_FIRST_PAGE = 6  # Chapter 4 starts on the page printed 6

SRC = pymupdf.open(REPORT)


# ------------------------------------------------------------------ helpers
def overlay(draw_fn):
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=letter)
    draw_fn(c)
    c.save()
    buf.seek(0)
    return PdfReader(buf).pages[0]


def baselines(pno, top, bottom):
    """Baselines (top-down points) of the original text lines starting between top and bottom."""
    ys = []
    for block in SRC[pno].get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            if top <= line["bbox"][1] <= bottom:
                ys.extend(span["origin"][1] for span in line["spans"])
    merged = []
    for y in sorted(ys):
        if not merged or y - merged[-1] > 1.0:
            merged.append(y)
    return merged


def wrap(text, first_width, width, size, font="LM"):
    lines, cur, limit = [], [], first_width
    for word in text.split():
        if not cur or pdfmetrics.stringWidth(" ".join(cur + [word]), font, size) <= limit:
            cur.append(word)
        else:
            lines.append(cur)
            cur, limit = [word], width
    lines.append(cur)
    return lines


def draw_justified(c, words, x0, x1, y, size, last=False, font="LM"):
    c.setFont(font, size)
    if last or len(words) == 1:
        c.drawString(x0, y, " ".join(words))
        return
    natural = sum(pdfmetrics.stringWidth(w, font, size) for w in words)
    gap = (x1 - x0 - natural) / (len(words) - 1)
    x = x0
    for w in words:
        c.drawString(x, y, w)
        x += pdfmetrics.stringWidth(w, font, size) + gap


def set_paragraph(c, text, ys, x0=72.0, x1=540.0, size=BODY, indent=0.0, justify=True):
    """Typeset text on exactly the given baselines (justified like LaTeX, last line ragged)."""
    lines = wrap(text, x1 - x0 - indent, x1 - x0, size)
    assert len(lines) == len(ys), f"{len(lines)} lines for {len(ys)} baselines: {text[:50]}"
    c.setFillColorRGB(0, 0, 0)
    for i, (words, y) in enumerate(zip(lines, ys)):
        start = x0 + (indent if i == 0 else 0)
        last = i == len(lines) - 1 or not justify
        draw_justified(c, words, start, x1, PAGE_H - y, size, last=last)


# --------------------------------------------------------- text corrections
# (page index, rectangle x0, top, x1, bottom in top-down points) to remove.
REDACT = {
    3: [
        (70, 180.0, 545, 263.8),     # 1.3.1 first paragraph
        (70, 265.4, 545, 320.8),     # 1.3.1 second paragraph
        (220.6, 389.0, 545, 402.8),  # 1.3.2 bullet: targets file
        (209.1, 429.8, 545, 443.4),  # 1.3.2 bullet: combo file
        (70, 446.6, 545, 516.4),     # 1.3.2 summary paragraph
        (70, 554.2, 545, 623.9),     # 1.3.3 paragraph
    ],
    4: [(70, 152.6, 545, 262.0)],    # Table 1.1 rows (rules kept)
    6: [
        (70, 334.6, 545, 376.0),     # end of Section 3.1's last paragraph
        (70, 678.6, 545, 716.0),     # Table 3.2, Tier 3 row (rules kept)
    ],
    7: [(0, 166, 612, 738)],         # Bibliography (re-typeset after Chapter 5)
}

# Section 1.3.1: the system ranks recorded effects and the LLM explains them;
# it does not predict, and it is a prototype, not a clinical framework.
OBJECTIVE_1 = (
    "Taking DDI-GPT as a baseline reference, this project develops an end-to-end decision-support "
    "prototype that screens and explains the side effects recorded for the drug pairs of a regimen "
    "of three or more drugs. It establishes an API-driven Graph-RAG pipeline over the Decagon "
    "biomedical graph. Instead of a heavy locally hosted pairwise model, a lightweight API-based "
    "LLM explains the retrieved evidence for a whole multi-drug regimen in a single request, in plain "
    "language and with low inference latency."
)
OBJECTIVE_2 = (
    "For a queried drug set, the system retrieves a local subgraph containing the drugs, their target "
    "proteins, protein–protein links among those targets, and known side-effect edges. It ranks the "
    "documented effects, serialises the evidence into compact context, and an API-based LLM writes "
    "a short natural-language explanation of the effects selected."
)
# Section 1.3.2: counts measured on the CSV files.
DATASET_PARA = (
    "In full, the graph contains 645 drugs, 19,089 proteins, 18,690 drug–protein edges, "
    "715,612 protein–protein edges, and 4,649,441 drug–drug side-effect records (63,473 drug pairs). "
    "Only 284 of the 645 drugs have known targets. Standard identifiers connect the three files "
    "without manual labelling or external scraping, so entities and relations map directly to typed "
    "nodes and edges. This makes a compact local subgraph available for Graph-RAG."
)
# Section 1.3.3: what is delivered.
DELIVERABLE = (
    "The deliverable is a functional, low-latency pipeline with a chat interface that accepts "
    "multi-drug queries, extracts a local subgraph, and returns ranked side effects with traceable "
    "evidence paths and short plain-language explanations. Training a graph neural network to predict "
    "novel unobserved adverse edges is outside the current phase. Continuous clinical variables—dose, "
    "pharmacokinetics, and renal/hepatic clearance—and wet-lab validation are also future work."
)
# Section 3.1, last three lines: the LLM no longer reasons over the context.
RETRIEVAL_END = (
    "two drugs remain in the neighbourhood and are covered by pairwise history. The assembled "
    "context is passed to the LLM, which only explains the effects already selected and ranked "
    "(Section 4.1.4)."
)
# Table 1.1: the chapters that exist.
T11_ROWS = [
    ("1", "Introduction", "Motivation, related work and gaps, objective and scope."),
    ("2", "Background", "Knowledge graphs, Graph-RAG, LLMs for biomedical prediction, and the formal problem statement."),
    ("3", "Methodology", "Multi-drug retrieval over the Decagon graph, pruning of noisy proteins, and tiered, traceable predictions."),
    ("4", "Experiments and Results", "Data and implementation, evaluation of faithfulness and of known interactions, and limitations."),
    ("5", "Conclusion and Future Work", "Outcomes and directions for extension."),
]
# Table 3.2, Tier 3 row: Tier 3 applies to a pair with no documented effect at
# all, and it is a hypothesis, not an effect "arising only from the regimen".
TIER3_ROW = (
    ["3: Hypothesised", "effect"],
    "A bridge links targets of two different drugs, and that pair has no documented side effect at all.",
    "A hypothesis for an untested pair, based only on protein links; it requires clinical validation.",
)


def draw_page3(c):
    set_paragraph(c, OBJECTIVE_1, baselines(3, 180, 256))
    set_paragraph(c, OBJECTIVE_2, baselines(3, 265, 312), indent=16.4)
    c.setFont("LM", BODY)
    c.drawString(220.1, PAGE_H - baselines(3, 389, 392)[0], ": human proteins targeted by 284 drugs (18,690 drug–protein edges).")
    c.drawString(208.6, PAGE_H - baselines(3, 430, 433)[0], ": observed side effects, 1,317 types, for co-prescribed drug pairs.")
    set_paragraph(c, DATASET_PARA, baselines(3, 446, 506))
    set_paragraph(c, DELIVERABLE, baselines(3, 554, 615))


def draw_page4(c):
    ys = iter(baselines(4, 155, 251))
    c.setFillColorRGB(0, 0, 0)
    for number, title, text in T11_ROWS:
        lines = wrap(text, 540 - 243.1, 540 - 243.1, 10.0)
        first = next(ys)
        c.setFont("LM", 10.0)
        c.drawString(72, PAGE_H - first, f"{number} — {title}")
        for k, words in enumerate(lines):
            c.drawString(243.1, PAGE_H - (first if k == 0 else next(ys)), " ".join(words))
    assert next(ys, None) is None, "Table 1.1 must keep its 8 text lines"


def draw_page6(c):
    set_paragraph(c, RETRIEVAL_END, baselines(6, 334, 366))
    ys = baselines(6, 679, 707)
    label, trigger, meaning = TIER3_ROW
    c.setFont("LM", TABLE)
    for text, y in zip(label, ys):
        c.drawString(72, PAGE_H - y, text)
    set_paragraph(c, trigger, ys, x0=172.9, x1=360.1, size=TABLE)
    set_paragraph(c, meaning, ys, x0=372.0, x1=540.0, size=TABLE)


# ------------------------------------------------------------ table of contents
def heading_pages(path):
    """First printed page on which each Chapter 4/5 heading appears."""
    found = {}
    for i, page in enumerate(PdfReader(path).pages):
        text = page.extract_text()
        for key in ("4.1.1", "4.1.2", "4.1.3", "4.1.4", "4.1.5", "4.2", "4.2.1", "4.2.2", "4.2.3", "4.2.4", "4.3"):
            if key not in found and re.search(r"^" + re.escape(key) + r"\s+[A-Z]", text, re.MULTILINE):
                found[key] = CH4_FIRST_PAGE + i
        if "5" not in found and re.search(r"^Conclusion and Future Work", text, re.MULTILINE):
            found["5"] = CH4_FIRST_PAGE + i
    return found


pages = heading_pages(CH4)
# Same layout as the Contents page: number columns at x=72/88.4/113.5, titles at
# 88.4/113.5/147.8, page numbers right-aligned at 540, rows 14.22pt apart and
# 25.1pt before a chapter row; 3.3's baseline is at y=371.8 (bottom-up).
TOC = [
    (72, 88.4, "4", "Experiments and Results", True, CH4_FIRST_PAGE),
    (88.4, 113.5, "4.1", "Implementation", False, CH4_FIRST_PAGE),
    (113.5, 147.8, "4.1.1", "Data", False, pages["4.1.1"]),
    (113.5, 147.8, "4.1.2", "Software Architecture", False, pages["4.1.2"]),
    (113.5, 147.8, "4.1.3", "Subgraph Retrieval, Pruning, and Ranking", False, pages["4.1.3"]),
    (113.5, 147.8, "4.1.4", "Large Language Model Usage", False, pages["4.1.4"]),
    (113.5, 147.8, "4.1.5", "Grounding, Validation, and Fallback Behaviour", False, pages["4.1.5"]),
    (88.4, 113.5, "4.2", "Evaluation", False, pages["4.2"]),
    (113.5, 147.8, "4.2.1", "What can and cannot be measured", False, pages["4.2.1"]),
    (113.5, 147.8, "4.2.2", "Faithfulness to the data", False, pages["4.2.2"]),
    (113.5, 147.8, "4.2.3", "Known interactions in larger regimens", False, pages["4.2.3"]),
    (113.5, 147.8, "4.2.4", "Tier 3 hypotheses", False, pages["4.2.4"]),
    (88.4, 113.5, "4.3", "Limitations", False, pages["4.3"]),
    (72, 88.4, "5", "Conclusion and Future Work", True, pages["5"]),
]


def leader(c, x0, x1, y, spacing=8.47):
    # Dots on the same 8.47pt grid as the original LaTeX leaders.
    c.setFont("LM", BODY)
    x = 72 + spacing * (int((x0 - 72) / spacing) + 1)
    while x <= x1:
        c.drawString(x, y, ".")
        x += spacing


def draw_toc(c):
    y = 371.8 - 25.1
    c.setFillColorRGB(0, 0, 0)
    for row, (num_x, title_x, num, title, bold, page) in enumerate(TOC):
        if bold and row:
            y -= 25.1 - 14.22
        font = "LM-Bold" if bold else "LM"
        c.setFont(font, BODY)
        c.drawString(num_x, y, num)
        c.drawString(title_x, y, title)
        if not bold:
            end = min(522, 540 - pdfmetrics.stringWidth(str(page), font, BODY) - 9)
            leader(c, title_x + pdfmetrics.stringWidth(title, font, BODY) + 3, end, y)
        c.setFont(font, BODY)
        c.drawRightString(540, y, str(page))
        y -= 14.22


# ------------------------------------------------------------------ assemble
def redacted_report():
    doc = pymupdf.open(REPORT)
    for pno, rects in REDACT.items():
        page = doc[pno]
        for rect in rects:
            page.add_redact_annot(pymupdf.Rect(*rect), fill=(1, 1, 1))
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                              graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)
    return PdfReader(io.BytesIO(doc.tobytes()))


main = redacted_report()
ch4 = PdfReader(CH4)
fixes = {1: draw_toc, 3: draw_page3, 4: draw_page4, 6: draw_page6}

writer = PdfWriter()
for i in range(7):
    page = main.pages[i]
    if i in fixes:
        page.merge_page(overlay(fixes[i]))
    writer.add_page(page)

section_end = main.pages[7]           # Section 3.3's end, header and printed "6" kept
section_end.merge_page(ch4.pages[0])  # Chapter 4 starts below it
writer.add_page(section_end)
for page in ch4.pages[1:]:
    writer.add_page(page)

with open(OUT, "wb") as f:
    writer.write(f)
print("wrote", OUT, "-", len(writer.pages), "pages; headings on", pages)
