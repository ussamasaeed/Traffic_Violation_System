import mysql.connector
from fpdf import FPDF
from datetime import datetime, timedelta
import os

# ============================
# MySQL Connection
# ============================
db = mysql.connector.connect(
    host="192.168.0.101",
    user="root",
    password="",
    database="traffic"
)

cursor = db.cursor(dictionary=True)

# ============================
# Paths (YOU WILL EDIT THESE)
# ============================
IMAGE1_BASE_PATH = "Z:\\img"     
IMAGE2_BASE_PATH = "Z:\\person"
OUTPUT_DIR       = "Z:\\generated_challans"  

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# Fetch challans without PDF
# ============================
cursor.execute("SELECT * FROM challan WHERE pdf_generated = 0")
challans = cursor.fetchall()

for c in challans:

    uid = c["uid"]

    # Build image paths
    img1_path = os.path.join(IMAGE1_BASE_PATH, f"{uid}.jpg")
    img2_path = os.path.join(IMAGE2_BASE_PATH, f"{uid}.jpg")

    # Load date and calculate due date
    challan_date = datetime.strptime(str(c["Date"]), "%Y-%m-%d")
    due_date = challan_date + timedelta(days=10)

    # ============================
    # Fine Logic
    # ============================
    violation = c["valiotionType"].lower()

    fine_amount = 0
    if "overspeed" in violation:
        fine_amount += 2000
    if "drift" in violation:
        fine_amount += 5000

    # Output PDF file
    pdf_filename = os.path.join(OUTPUT_DIR, f"challan_{uid}.pdf")

    pdf = FPDF()
    pdf.add_page()

    # ===================================
    # Title
    # ===================================
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "Traffic Challan", ln=True, align="C")
    pdf.ln(10)

    # ===================================
    # Add Images (handles missing images)
    # ===================================
    y_position = 30

    if os.path.exists(img1_path):
        pdf.image(img1_path, x=10, y=y_position, w=90)
    else:
        pdf.set_font("Arial", "B", 12)
        pdf.text(10, y_position + 10, "Image 1 Not Found")

    if os.path.exists(img2_path):
        pdf.image(img2_path, x=110, y=y_position, w=90)
    else:
        pdf.set_font("Arial", "B", 12)
        pdf.text(110, y_position + 10, "Image 2 Not Found")

    pdf.ln(95)

    # ===================================
    # Challan Details
    # ===================================
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, f"Challan UID: {c['uid']}", ln=True)
    pdf.cell(0, 10, f"License Plate: {c['LicensePlate']}", ln=True)
    pdf.cell(0, 10, f"Vehicle Type: {c['Type']}", ln=True)
    pdf.cell(0, 10, f"Violation Type: {c['valiotionType']}", ln=True)
    pdf.cell(0, 10, f"Speed: {c['speed']} km/h", ln=True)

    pdf.cell(0, 10, f"Date: {c['Date']}", ln=True)
    pdf.cell(0, 10, f"Time: {c['Time']}", ln=True)

    pdf.cell(0, 10, f"Fine Amount: Rs {fine_amount}", ln=True)

    # ===================================
    # Due Date
    # ===================================
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Due Date: {due_date.strftime('%Y-%m-%d')}", ln=True)

    # Save PDF
    pdf.output(pdf_filename)
    print(f"Generated PDF: {pdf_filename}")

    # ===================================
    # SQL UPDATE
    # Safe update — includes FineAmount only if column exists
    # ===================================

    try:
        cursor.execute("""
            UPDATE challan 
            SET pdf_generated = 1, FineAmount = %s 
            WHERE uid = %s
        """, (fine_amount, uid))
    except:
        # If FineAmount column does NOT exist → update without it
        cursor.execute("""
            UPDATE challan 
            SET pdf_generated = 1 
            WHERE uid = %s
        """, (uid,))

    db.commit()

print("All new challans processed!")
