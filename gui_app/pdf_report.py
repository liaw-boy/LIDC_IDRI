"""Generate a one-page clinical-style PDF report for a Lung Nodule analysis.

Layout:
  Title bar | patient info | KPI strip | Lung-RADS table | inline screenshots | footer
"""
from __future__ import annotations

import datetime
import os
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# Lung-RADS band -> background color
_BAND_COLOR = {
    "2":  colors.HexColor("#16a34a"),
    "3":  colors.HexColor("#2563eb"),
    "4A": colors.HexColor("#eab308"),
    "4B": colors.HexColor("#f97316"),
    "4X": colors.HexColor("#dc2626"),
}


def _register_fonts() -> str:
    """Register a CJK-capable TTF font, fall back to Helvetica if unavailable."""
    # (path, subfontIndex_or_None) — index needed for .ttc collections
    candidates = [
        ("/mnt/windows_data_C/Windows/Fonts/NotoSansTC-VF.ttf", None),
        ("/mnt/windows_data_C/Windows/Fonts/NotoSansHK-VF.ttf", None),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 1),
        ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", None),
    ]
    for path, subindex in candidates:
        if not os.path.exists(path):
            continue
        try:
            if subindex is None:
                pdfmetrics.registerFont(TTFont("ReportFont", path))
            else:
                pdfmetrics.registerFont(TTFont("ReportFont", path, subfontIndex=subindex))
            return "ReportFont"
        except Exception:
            continue
    return "Helvetica"


def generate_pdf_report(
    output_path: str,
    patient_id: str,
    nodules: list[dict],
    screenshots: Iterable[str] = (),
    kpi: dict | None = None,
) -> str:
    """Render a one-page PDF report.

    nodules: list of dicts with keys idx, lung_rads, label, mal_prob, action, n_slices
    screenshots: iterable of PNG paths to include inline
    kpi: optional dict with keys recall, fpr, f1, n_test
    """
    font = _register_fonts()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=12 * mm,  bottomMargin=12 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "title", parent=styles["Title"],
        fontName=font, fontSize=18, leading=22,
        textColor=colors.HexColor("#0f172a"),
    )
    h2_style = ParagraphStyle(
        "h2", parent=styles["Heading2"],
        fontName=font, fontSize=12, leading=16,
        textColor=colors.HexColor("#1e293b"),
    )
    body_style = ParagraphStyle(
        "body", parent=styles["BodyText"],
        fontName=font, fontSize=9, leading=13,
        textColor=colors.HexColor("#334155"),
    )
    meta_style = ParagraphStyle(
        "meta", parent=body_style, fontSize=8, textColor=colors.HexColor("#64748b"),
    )

    story: list = []

    story.append(Paragraph("Lung Nodule AI 診斷報告", title_style))
    story.append(Paragraph(
        f"病患 ID: <b>{patient_id}</b>　|　報告生成: "
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        meta_style,
    ))
    story.append(Spacer(1, 6))

    # KPI strip
    if kpi:
        kpi_data = [[
            Paragraph(f"<b>Test Set</b><br/>n={kpi.get('n_test', '—')}", body_style),
            Paragraph(f"<b>Malignant Recall</b><br/>{kpi.get('recall', 0)*100:.1f}%", body_style),
            Paragraph(f"<b>Benign FPR</b><br/>{kpi.get('fpr', 0)*100:.1f}%", body_style),
            Paragraph(f"<b>F1 Score</b><br/>{kpi.get('f1', 0):.3f}", body_style),
        ]]
        kpi_table = Table(kpi_data, colWidths=[42*mm]*4)
        kpi_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
            ("BOX",        (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID",  (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 8))

    # Lung-RADS findings table
    story.append(Paragraph("結節評估結果", h2_style))
    header = ["#", "Lung-RADS", "判斷", "惡性機率", "聚合切片", "建議處置"]
    rows = [header]
    for n in sorted(nodules, key=lambda x: -x["mal_prob"]):
        rows.append([
            str(n["idx"]),
            n["lung_rads"],
            n["label"],
            f"{n['mal_prob']*100:.1f}%",
            f"{n['n_slices']}",
            n["action"],
        ])
    findings = Table(rows, colWidths=[10*mm, 22*mm, 25*mm, 24*mm, 22*mm, 65*mm])
    style_cmds = [
        ("FONTNAME",   (0, 0), (-1, -1), font),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0),  colors.HexColor("#1e293b")),
        ("TEXTCOLOR",  (0, 0), (-1, 0),  colors.white),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("ALIGN",      (5, 1), (5, -1),  "LEFT"),
        ("INNERGRID",  (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("BOX",        (0, 0), (-1, -1), 0.5,  colors.HexColor("#94a3b8")),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    # Color the Lung-RADS band cell per row
    for i, n in enumerate(sorted(nodules, key=lambda x: -x["mal_prob"]), start=1):
        band_color = _BAND_COLOR.get(n["lung_rads"], colors.HexColor("#64748b"))
        style_cmds.append(("BACKGROUND", (1, i), (1, i), band_color))
        style_cmds.append(("TEXTCOLOR",  (1, i), (1, i), colors.white))
    findings.setStyle(TableStyle(style_cmds))
    story.append(findings)
    story.append(Spacer(1, 10))

    # Screenshots in a row, scaled to fit
    valid_shots = [s for s in screenshots if s and os.path.exists(s)]
    if valid_shots:
        story.append(Paragraph("AI 推論視覺化", h2_style))
        thumb_w = (180 * mm) / max(len(valid_shots), 1)
        thumb_h = thumb_w * 0.62
        thumb_row = [Image(s, width=thumb_w, height=thumb_h) for s in valid_shots[:3]]
        thumb_table = Table([thumb_row])
        thumb_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(thumb_table)
        story.append(Spacer(1, 6))

    # Footer
    story.append(Paragraph(
        "資料來源: LIDC-IDRI 公開資料集（51 病患 holdout test split, seed=42）。"
        " 模型: YOLO11n + NoduleClassifier (CBAM + Attribute Feedback)。"
        " 此報告為 AI 輔助診斷工具，最終診斷需由放射科醫師判讀。",
        meta_style,
    ))

    doc.build(story)
    return output_path
