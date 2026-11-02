"""Generate the sample PDF: a fictional 6-page employee handbook with facts spread across pages,
so citation correctness is easy to check. Run: python scripts/make_sample_pdf.py"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

PAGES = [
    ("Northwind Robotics Employee Handbook", [
        "Welcome to Northwind Robotics. This handbook explains the policies that apply to all full-time employees in the United States.",
        "Northwind Robotics designs warehouse automation systems and was founded in 2017 in Pittsburgh, Pennsylvania. The company has offices in Pittsburgh, Austin and Toronto.",
        "Questions about this handbook should go to the People Operations team at people@northwind.example.",
    ]),
    ("Time Off", [
        "Full-time employees receive 20 days of paid time off (PTO) per calendar year, accrued monthly.",
        "Up to 5 unused PTO days can be carried over into the next year. Carried-over days expire on March 31.",
        "Northwind observes 11 paid company holidays, including a winter shutdown between December 24 and January 1.",
        "Sick leave is separate from PTO: employees get 10 paid sick days per year, and a doctor's note is required only after 3 consecutive sick days.",
    ]),
    ("Remote and Hybrid Work", [
        "Northwind follows a hybrid model. Employees assigned to an office are expected on site on Tuesdays and Thursdays.",
        "Fully remote roles must be approved by a department head and People Operations.",
        "Remote employees receive a one-time home office stipend of $750 and a monthly internet allowance of $60.",
        "Employees may work from another country for up to 30 days per year with manager approval and a tax review.",
    ]),
    ("Expenses and Travel", [
        "Business expenses must be submitted in the expense tool within 30 days of purchase, with an itemized receipt for anything over $25.",
        "The daily meal limit during business travel is $75 in the United States and $90 internationally.",
        "Economy class is required for flights under 6 hours. Premium economy is allowed for flights of 6 hours or longer.",
        "Hotel bookings should stay under $250 per night unless the city is on the high-cost list maintained by Finance.",
    ]),
    ("Learning and Development", [
        "Each employee has an annual learning budget of $2,000 for courses, conferences, certifications and books.",
        "Employees can take 3 paid learning days per year to attend training or study for a certification.",
        "Tuition reimbursement for degree programs requires 12 months of tenure and approval from a director.",
    ]),
    ("Security and Equipment", [
        "All laptops use full-disk encryption and must lock automatically after 5 minutes of inactivity.",
        "Multi-factor authentication is required for email, source code and the HR system.",
        "Report a lost or stolen device to IT Security within 1 hour by emailing security@northwind.example.",
        "Customer data must never be copied to personal devices or uploaded to unapproved AI tools.",
    ]),
]


def build(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = []
    for i, (heading, paragraphs) in enumerate(PAGES):
        story.append(Paragraph(heading, styles["Title" if i == 0 else "Heading1"]))
        story.append(Spacer(1, 12))
        for text in paragraphs:
            story.append(Paragraph(text, styles["BodyText"]))
            story.append(Spacer(1, 8))
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(path), pagesize=LETTER, title="Northwind Robotics Employee Handbook").build(story)
    return path


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data/sample/northwind_handbook.pdf"
    print(f"Wrote {build(out)}")
