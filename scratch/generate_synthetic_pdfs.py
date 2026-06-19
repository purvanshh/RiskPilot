import os

from reportlab.pdfgen import canvas


def generate_pdf(path: str, title: str, lines: list[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    c = canvas.Canvas(path)

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, title)

    # Body
    c.setFont("Helvetica", 12)
    y = 700
    for line in lines:
        c.drawString(100, y, line)
        y -= 25
        if y < 50:
            c.showPage()
            y = 750
            c.setFont("Helvetica", 12)

    c.save()
    print(f"Generated PDF: {path}")


def main():
    # 1. Alice Johnson (APP-001)
    generate_pdf(
        "data/synthetic_docs/APP-001-id.pdf",
        "Identity Document",
        [
            "Document Type: ID Proof",
            "Name: Alice Johnson",
            "DOB: 12/10/1990",
            "ID Number: ID-908127391",
        ],
    )
    generate_pdf(
        "data/synthetic_docs/APP-001-bank_statement.pdf",
        "Bank Account Statement",
        [
            "Account Holder: Alice Johnson",
            "Statement Period: May 2026",
            "Monthly deposit: $6,666",
            "Balance: $12,500",
            "Monthly debt: $1,200",
        ],
    )
    generate_pdf(
        "data/synthetic_docs/APP-001-pay_slip.pdf",
        "Pay Slip",
        [
            "Employee Name: Alice Johnson",
            "Employer: TechCorp",
            "Pay Period: Monthly",
            "Gross pay: $6,666",
            "Net Pay: $5,120",
        ],
    )
    generate_pdf(
        "data/synthetic_docs/APP-001-employment_letter.pdf",
        "Employment Verification Letter",
        [
            "To Whom It May Concern,",
            "Employment confirmation: Alice Johnson has been employed at TechCorp for 3 years.",
            "Position: Senior Software Engineer",
            "Status: Full-time",
        ],
    )

    # 2. Bob Smith (APP-002)
    generate_pdf(
        "data/synthetic_docs/APP-002-id.pdf",
        "Identity Document",
        [
            "Document Type: ID Proof",
            "Name: Bob Smith",
            "DOB: 05/04/1985",
            "ID Number: ID-850405123",
        ],
    )
    generate_pdf(
        "data/synthetic_docs/APP-002-bank_statement.pdf",
        "Bank Account Statement",
        [
            "Account Holder: Bob Smith",
            "Statement Period: May 2026",
            "Monthly deposit: $2,500",
            "Balance: $1,200",
            "Monthly debt: $1,375",
        ],
    )
    generate_pdf(
        "data/synthetic_docs/APP-002-pay_slip.pdf",
        "Pay Slip",
        [
            "Employee Name: Bob Smith",
            "Employer: RetailInc",
            "Pay Period: Monthly",
            "Gross pay: $2,500",
        ],
    )

    # 3. Charlie Brown (APP-003)
    generate_pdf(
        "data/synthetic_docs/APP-003-id.pdf",
        "Identity Document",
        ["Document Type: ID Proof", "Name: Charlie Brown", "DOB: 09/09/1995"],
    )
    generate_pdf(
        "data/synthetic_docs/APP-003-bank_statement.pdf",
        "Bank Account Statement",
        ["Account Holder: Charlie Brown", "Monthly deposit: $5,000", "Monthly debt: $1,750"],
    )
    generate_pdf(
        "data/synthetic_docs/APP-003-pay_slip.pdf",
        "Pay Slip",
        ["Employee Name: Charlie Brown", "Employer: BuildCorp", "Gross pay: $5,000"],
    )
    generate_pdf(
        "data/synthetic_docs/APP-003-employment_letter.pdf",
        "Employment Verification Letter",
        [
            "Employment confirmation: Charlie Brown has been employed for 2 years.",
            "Employer: BuildCorp",
        ],
    )

    # 4. Diana Prince (APP-004)
    generate_pdf(
        "data/synthetic_docs/APP-004-id.pdf",
        "Identity Document",
        ["Document Type: ID Proof", "Name: Diana Prince", "DOB: 01/01/1980"],
    )
    generate_pdf(
        "data/synthetic_docs/APP-004-pay_slip.pdf",
        "Pay Slip",
        ["Employee Name: Diana Prince", "Employer: JusticeLeague", "Gross pay: $7,500"],
    )

    # 5. Evan Wright (APP-005)
    generate_pdf(
        "data/synthetic_docs/APP-005-id.pdf",
        "Identity Document",
        ["Document Type: ID Proof", "Name: Evan Wright", "DOB: 02/02/1992"],
    )
    generate_pdf(
        "data/synthetic_docs/APP-005-bank_statement.pdf",
        "Bank Account Statement",
        ["Account Holder: Evan Wright", "Monthly deposit: $8,333", "Monthly debt: $3,166"],
    )
    generate_pdf(
        "data/synthetic_docs/APP-005-pay_slip.pdf",
        "Pay Slip",
        ["Employee Name: Evan Wright", "Employer: DesignStudio", "Gross pay: $8,333"],
    )
    generate_pdf(
        "data/synthetic_docs/APP-005-employment_letter.pdf",
        "Employment Verification Letter",
        ["Employment confirmation: Evan has been employed at DesignStudio for 11 months."],
    )

    # 6. Frank Forger (APP-006)
    generate_pdf(
        "data/synthetic_docs/APP-006-id.pdf",
        "Identity Document",
        ["Document Type: ID Proof", "Name: Frank Forger", "DOB: 03/12/1988"],
    )
    generate_pdf(
        "data/synthetic_docs/APP-006-pay_slip.pdf",
        "Pay Slip",
        ["Employee Name: Francis Forgett", "Employer: GlobalFinance", "Pay: $7,000/mo."],
    )
    generate_pdf(
        "data/synthetic_docs/APP-006-bank_statement.pdf",
        "Bank Account Statement",
        ["Account Holder: F. Forger", "Monthly deposit: $7,000"],
    )


if __name__ == "__main__":
    main()
