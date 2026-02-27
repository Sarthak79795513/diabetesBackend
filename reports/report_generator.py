# backend/reports/report_generator.py

from fpdf import FPDF
import os
from models.patient import Patient

class ReportGenerator:
    def __init__(self):
        # Attributes (as per class diagram)
        self.reportFormat = "PDF"
        self.reportData = {}

        # reports folder auto-create
        if not os.path.exists("generated_reports"):
            os.makedirs("generated_reports")

    # +generatePatientReport(patient : Patient, riskLevel : String) : void
    def generatePatientReport(self, patient: Patient, riskLevel: str):
        self.reportData = {
            "Name": patient.name,
            "Age": patient.age,
            "Gender": patient.gender,
            "Pregnancies": patient.pregnancies,
            "Glucose": patient.glucose,
            "Blood Pressure": patient.bloodPressure,
            "Skin Thickness": patient.skinThickness,
            "Insulin": patient.insulin,
            "BMI": patient.BMI,
            "Diabetes Pedigree Function": patient.diabetesPedigreeFunction,
            "Risk Level": riskLevel
        }

    # +exportReport(format : String) : void
    def exportReport(self, format: str = "PDF"):
        if format.upper() == "PDF":
            self._exportPDF()
        elif format.upper() == "HTML":
            self._exportHTML()
        else:
            print("Unsupported report format")

    # ---------------- INTERNAL METHODS ---------------- #

    def _exportPDF(self):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="Diabetes Risk Prediction Report", ln=True, align="C")
        pdf.ln(10)

        for key, value in self.reportData.items():
            pdf.cell(200, 8, txt=f"{key}: {value}", ln=True)

        filename = "generated_reports/diabetes_report.pdf"
        pdf.output(filename)

        print(f"PDF Report Generated: {filename}")

    def _exportHTML(self):
        html_content = "<html><head><title>Diabetes Report</title></head><body>"
        html_content += "<h2>Diabetes Risk Prediction Report</h2><table border='1'>"

        for key, value in self.reportData.items():
            html_content += f"<tr><td>{key}</td><td>{value}</td></tr>"

        html_content += "</table></body></html>"

        filename = "generated_reports/diabetes_report.html"
        with open(filename, "w") as file:
            file.write(html_content)

        print(f"HTML Report Generated: {filename}")
