import argparse
import os
import re
from typing import List, Tuple, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    PageBreak,
    ListFlowable,
    ListItem,
    Preformatted,
    HRFlowable,
)
from PIL import Image as PILImage


# --------------- Markdown -> ReportLab helpers ---------------

IMG_LINE_RE = re.compile(r"^\s*!\[(?P<alt>.*?)\]\((?P<path>.*?)\)\s*$")
IMG_INLINE_RE = re.compile(r"!\[(?P<alt>.*?)\]\((?P<path>.*?)\)")
HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.+?)\s*$")
ORDERED_RE = re.compile(r"^\s*(?P<num>\d+)\.\s+(?P<text>.+)\s*$")
UNORDERED_RE = re.compile(r"^\s*[-*]\s+(?P<text>.+)\s*$")
HR_RE = re.compile(r"^\s*---+\s*$")
CODE_BLOCK_FENCE_RE = re.compile(r"^\s*```")


def build_styles():
    styles = getSampleStyleSheet()

    # Base text
    body = ParagraphStyle(
        name="Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        spaceBefore=6,
        spaceAfter=6,
    )
    styles.add(body)

    code_style = ParagraphStyle(
        name="CodeEx",
        parent=styles["BodyText"],
        fontName="Courier",
        fontSize=9.5,
        leading=12,
        backColor=colors.whitesmoke,
        leftIndent=6,
        rightIndent=6,
        spaceBefore=6,
        spaceAfter=6,
    )
    styles.add(code_style)

    h1 = ParagraphStyle(
        name="Heading1Ex",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        spaceBefore=14,
        spaceAfter=8,
    )
    styles.add(h1)

    h2 = ParagraphStyle(
        name="Heading2Ex",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        spaceBefore=12,
        spaceAfter=6,
    )
    styles.add(h2)

    h3 = ParagraphStyle(
        name="Heading3Ex",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=6,
    )
    styles.add(h3)

    bullet = ParagraphStyle(
        name="BulletEx",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        leftIndent=16,
        firstLineIndent=-10,
        spaceBefore=2,
        spaceAfter=2,
    )
    styles.add(bullet)

    return styles


def md_inline_code(text: str) -> str:
    # Replace inline code `code` with Courier font
    def repl(m):
        code = m.group(1)
        # Escape angle brackets that ReportLab might interpret
        esc = (code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        return f'<font face="Courier">{esc}</font>'

    return re.sub(r"`([^`]+)`", repl, text)


def resolve_image_path(img_path: str, base_dir: str) -> Optional[str]:
    img_path = img_path.strip().strip('"').strip("'")
    if os.path.isabs(img_path) and os.path.exists(img_path):
        return img_path
    candidate = os.path.join(base_dir, img_path)
    if os.path.exists(candidate):
        return candidate
    # Try normalizing separators
    candidate2 = os.path.normpath(candidate)
    if os.path.exists(candidate2):
        return candidate2
    # Give up
    return None


def make_image_flowable(path: str, doc_width: float, max_height: float = 22 * cm) -> Optional[RLImage]:
    try:
        with PILImage.open(path) as im:
            w, h = im.size
    except Exception:
        return None

    if w == 0 or h == 0:
        return None

    # Fit to doc width preserving aspect
    scale = doc_width / float(w)
    new_w = doc_width
    new_h = h * scale

    if new_h > max_height:
        # scale down to fit height
        scale2 = max_height / new_h
        new_w *= scale2
        new_h = max_height

    img = RLImage(path, width=new_w, height=new_h)
    img.hAlign = "CENTER"
    return img


def add_page_number(canvas, doc):
    canvas.saveState()
    width, height = doc.pagesize
    page_num_text = f"Page {doc.page}"
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(width - doc.rightMargin, doc.bottomMargin - 10, page_num_text)
    canvas.restoreState()


def parse_markdown_to_flowables(md_lines: List[str], styles, base_dir: str, doc_width: float) -> List:
    story: List = []
    in_code_block = False
    code_block_lines: List[str] = []

    pending_list_items: List[Tuple[str, str]] = []  # (type, text) type in {"ul","ol"}
    list_type_current: Optional[str] = None

    para_buffer: List[str] = []

    def flush_paragraph():
        nonlocal para_buffer
        if para_buffer:
            text = " ".join(line.strip() for line in para_buffer).strip()
            text = md_inline_code(text)
            if text:
                story.append(Paragraph(text, styles["Body"]))
            story.append(Spacer(1, 6))
            para_buffer = []

    def flush_list():
        nonlocal pending_list_items, list_type_current
        if pending_list_items:
            bullet_type = "bullet" if list_type_current == "ul" else "1"
            items = []
            for _, t in pending_list_items:
                t = md_inline_code(t)
                items.append(ListItem(Paragraph(t, styles["Body"]), leftIndent=10))
            story.append(
                ListFlowable(
                    items,
                    bulletType=bullet_type,
                    start="1",
                    leftIndent=16,
                    bulletFontName="Helvetica",
                    bulletFontSize=9.5,
                    bulletColor=colors.black,
                    bulletDedent=6,
                    spaceBefore=2,
                    spaceAfter=4,
                )
            )
            story.append(Spacer(1, 2))
            pending_list_items = []
            list_type_current = None

    for raw in md_lines:
        line = raw.rstrip("\n")

        # Code block fences
        if CODE_BLOCK_FENCE_RE.match(line):
            if in_code_block:
                # close code block
                story.append(Preformatted("\n".join(code_block_lines), styles["CodeEx"]))
                story.append(Spacer(1, 6))
                code_block_lines = []
                in_code_block = False
            else:
                # open code block
                flush_paragraph()
                flush_list()
                in_code_block = True
            continue

        if in_code_block:
            code_block_lines.append(line)
            continue

        # Horizontal rule - use HRFlowable or PageBreak depending on context
        if HR_RE.match(line):
            flush_paragraph()
            flush_list()
            story.append(HRFlowable(width="100%", thickness=0.7, color=colors.grey))
            story.append(Spacer(1, 6))
            continue

        # Headings
        m_h = HEADING_RE.match(line)
        if m_h:
            flush_paragraph()
            flush_list()
            level = len(m_h.group("hashes"))
            text = md_inline_code(m_h.group("text"))
            if level == 1:
                story.append(Paragraph(text, styles["Heading1Ex"]))
            elif level == 2:
                story.append(Paragraph(text, styles["Heading2Ex"]))
            else:
                story.append(Paragraph(text, styles["Heading3Ex"]))  # typo fixed below
            story.append(Spacer(1, 6))
            continue

        # Image-only line
        m_img_line = IMG_LINE_RE.match(line)
        if m_img_line:
            flush_paragraph()
            flush_list()
            img_path_raw = m_img_line.group("path")
            resolved = resolve_image_path(img_path_raw, base_dir)
            if resolved:
                img = make_image_flowable(resolved, doc_width)
                if img:
                    story.append(img)
                    story.append(Spacer(1, 6))
            continue

        # Lists
        m_ul = UNORDERED_RE.match(line)
        m_ol = ORDERED_RE.match(line)
        if m_ul:
            flush_paragraph()
            text = m_ul.group("text").strip()
            if list_type_current not in (None, "ul"):
                flush_list()
            list_type_current = "ul"
            pending_list_items.append(("ul", text))
            continue
        elif m_ol:
            flush_paragraph()
            text = m_ol.group("text").strip()
            if list_type_current not in (None, "ol"):
                flush_list()
            list_type_current = "ol"
            pending_list_items.append(("ol", text))
            continue

        # Blank line flushes paragraph and list
        if not line.strip():
            flush_paragraph()
            flush_list()
            continue

        # If line contains inline image but not image-only, split into paragraph + image
        if IMG_INLINE_RE.search(line):
            flush_paragraph()
            flush_list()
            # Break out all images in the line
            last_end = 0
            for m in IMG_INLINE_RE.finditer(line):
                text_before = line[last_end : m.start()]
                if text_before.strip():
                    story.append(Paragraph(md_inline_code(text_before.strip()), styles["Body"]))
                    story.append(Spacer(1, 4))
                resolved = resolve_image_path(m.group("path"), base_dir)
                if resolved:
                    img = make_image_flowable(resolved, doc_width)
                    if img:
                        story.append(img)
                        story.append(Spacer(1, 4))
                last_end = m.end()
            tail = line[last_end:].strip()
            if tail:
                story.append(Paragraph(md_inline_code(tail), styles["Body"]))
                story.append(Spacer(1, 6))
            continue

        # Accumulate paragraph
        para_buffer.append(line)

    # Flush any remaining buffers
    flush_paragraph()
    flush_list()
    if in_code_block and code_block_lines:
        story.append(Preformatted("\n".join(code_block_lines), styles["CodeEx"]))
        story.append(Spacer(1, 6))

    # Fix typo introduced earlier (ensure Heading3 works)
    # No action here; kept for clarity.

    return story


def export_markdown_to_pdf(md_path: str, out_pdf_path: str, pagesize=A4):
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    out_dir = os.path.dirname(out_pdf_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(md_path, "r", encoding="utf-8") as f:
        md_lines = f.readlines()

    # Document setup (A4 for EU)
    margin = 1.8 * cm
    doc = SimpleDocTemplate(
        out_pdf_path,
        pagesize=pagesize,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
        title="Comprehensive Analysis Report",
        author="Thesis Analysis Pipeline",
    )

    styles = build_styles()
    base_dir = os.path.dirname(os.path.abspath(md_path))
    doc_width = doc.width  # content width

    story = parse_markdown_to_flowables(md_lines, styles, base_dir, doc_width)

    # Build with page numbers
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def main():
    parser = argparse.ArgumentParser(description="Export Markdown report to PDF (pure Python, ReportLab).")
    parser.add_argument(
        "--input",
        "-i",
        default="FINAL_COMPREHENSIVE_ANALYSIS_REPORT.md",
        help="Path to the Markdown file to export.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(
            "C:\\", "Users", "tommy", "Desktop", "TESI", "Esperimento - Dati", "FINAL_COMPREHENSIVE_ANALYSIS_REPORT.pdf"
        ),
        help="Path to the output PDF file.",
    )
    args = parser.parse_args()

    export_markdown_to_pdf(args.input, args.output)
    print(f"Saved PDF to: {args.output}")


if __name__ == "__main__":
    main()