import re
from pathlib import Path


def _escape_pdf_text(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_lines(text: str, width: int = 95):
    out = []
    for line in text.splitlines():
        if not line.strip():
            out.append("")
            continue
        # Keep headings/bullets readable.
        prefix = ""
        m = re.match(r"^(\s*[-#]+ ?)", line)
        if m:
            prefix = m.group(1)
            body = line[len(prefix) :]
        else:
            body = line

        words = body.split()
        cur = prefix
        for w in words:
            if len(cur) + (1 if cur.strip() else 0) + len(w) > width:
                out.append(cur.rstrip())
                cur = (prefix if prefix else "") + w
            else:
                cur = (cur + (" " if cur.strip() else "") + w).rstrip()
        if cur.strip():
            out.append(cur.rstrip())
    return out


def _build_pdf_pages(lines, font_size=11, leading=14, margin_x=48, margin_y=54):
    # US Letter: 612x792 points
    page_w, page_h = 612, 792
    usable_h = page_h - 2 * margin_y
    lines_per_page = max(1, int(usable_h // leading))

    pages = []
    for i in range(0, len(lines), lines_per_page):
        chunk = lines[i : i + lines_per_page]
        pages.append(chunk)
    return pages


def write_simple_pdf(text: str, out_path: Path):
    lines = _wrap_lines(text)
    pages = _build_pdf_pages(lines)

    objects = []

    def add_obj(payload: str) -> int:
        objects.append(payload)
        return len(objects)

    # 1: Catalog, 2: Pages, 3..: page + content, font as shared object
    font_obj = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_objs = []
    content_objs = []

    for page_lines in pages:
        # Content stream
        y_start = 792 - 54  # top margin
        content = []
        content.append("BT")
        content.append(f"/F1 11 Tf")
        content.append(f"48 {y_start} Td")
        content.append("14 TL")
        for ln in page_lines:
            if ln == "":
                content.append("T*")
                continue
            content.append(f"({_escape_pdf_text(ln)}) Tj")
            content.append("T*")
        content.append("ET")
        stream = "\n".join(content).encode("utf-8")
        content_obj = add_obj(f"<< /Length {len(stream)} >>\nstream\n{stream.decode('utf-8')}\nendstream")
        content_objs.append(content_obj)

        page_obj = add_obj(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> "
            f"/Contents {content_obj} 0 R >>"
        )
        page_objs.append(page_obj)

    kids = " ".join(f"{p} 0 R" for p in page_objs)
    pages_obj = add_obj(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_objs)} >>")
    # Catalog references pages object number (which is last object index right now)
    catalog_obj = add_obj(f"<< /Type /Catalog /Pages {pages_obj} 0 R >>")

    # Build xref
    xref_offsets = []
    header = "%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    body = ""
    offset = len(header.encode("latin-1"))
    for i, obj in enumerate(objects, start=1):
        xref_offsets.append(offset)
        chunk = f"{i} 0 obj\n{obj}\nendobj\n"
        body += chunk
        offset += len(chunk.encode("utf-8"))

    xref_start = offset
    xref = ["xref", f"0 {len(objects)+1}", "0000000000 65535 f "]
    for off in xref_offsets:
        xref.append(f"{off:010d} 00000 n ")
    xref_txt = "\n".join(xref) + "\n"
    trailer = (
        "trailer\n"
        f"<< /Size {len(objects)+1} /Root {catalog_obj} 0 R >>\n"
        "startxref\n"
        f"{xref_start}\n"
        "%%EOF\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(header.encode("latin-1") + body.encode("utf-8") + xref_txt.encode("utf-8") + trailer.encode("utf-8"))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate a simple PDF from a Markdown/text report.")
    p.add_argument("--in", dest="in_path", default=str(Path(__file__).parent / "phase1_report.md"))
    p.add_argument("--out", dest="out_path", default=str(Path(__file__).parent / "phase1_report.pdf"))
    args = p.parse_args()

    md_path = Path(args.in_path)
    pdf_path = Path(args.out_path)
    text = md_path.read_text(encoding="utf-8")
    write_simple_pdf(text, pdf_path)
    print(str(pdf_path))
