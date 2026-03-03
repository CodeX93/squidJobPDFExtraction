from reportlab.pdfgen import canvas

def create_pdf(filename):
    c = canvas.Canvas(filename)
    c.drawString(100, 750, "BID DOCUMENT")
    c.drawString(100, 730, "Bid Number: GEM/2026/B/1234567")
    c.drawString(100, 710, "Dated: 15-02-2026")
    c.drawString(100, 690, "Ministry: Ministry of Finance")
    c.drawString(100, 670, "Department: Department of Expenditure")
    c.drawString(100, 650, "Item: Desktop Computers")
    c.drawString(100, 630, "Quantity: 50")
    c.save()

if __name__ == "__main__":
    create_pdf("sample.pdf")
