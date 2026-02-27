import sqlite3
import os

class Database:
    def __init__(self):
        # backend folder ka path
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DB_PATH = os.path.join(BASE_DIR, "diabetes.db")

        self.connection = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.cursor = self.connection.cursor()

        self.createTable()

    def createTable(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                age INTEGER,
                gender TEXT,
                pregnancies INTEGER,
                glucose REAL,
                bloodPressure REAL,
                skinThickness REAL,
                insulin REAL,
                BMI REAL,
                diabetesPedigreeFunction REAL,
                riskLevel TEXT
            )
        """)
        self.connection.commit()

    def savePatientData(self, patient, riskLevel):
        self.cursor.execute("""
            INSERT INTO patients (
                name, age, gender, pregnancies, glucose,
                bloodPressure, skinThickness, insulin,
                BMI, diabetesPedigreeFunction, riskLevel
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            patient.name,
            patient.age,
            patient.gender,
            patient.pregnancies,
            patient.glucose,
            patient.bloodPressure,
            patient.skinThickness,
            patient.insulin,
            patient.BMI,
            patient.diabetesPedigreeFunction,
            riskLevel
        ))
        self.connection.commit()
