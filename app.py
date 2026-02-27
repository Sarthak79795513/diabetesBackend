import logging
import os
import sqlite3
import traceback

import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ---------------- PROJECT IMPORTS ----------------
from database.database_manager import Database
from models.patient import Patient
from reports.report_generator import ReportGenerator

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- FLASK APP ----------------
app = Flask(__name__, static_folder="../project/dist", static_url_path="/")

CORS_ORIGIN = os.environ.get("CORS_ORIGIN", "*")
CORS(app, resources={r"/*": {"origins": CORS_ORIGIN}}, supports_credentials=True)

logger.info("Diabetes Risk Prediction API Started")

# ---------------- DATABASE ----------------
def init_user_table():
    conn = sqlite3.connect("diabetes.db")
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        email TEXT,
        full_name TEXT,
        phone TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Migration for existing databases
    for col in ['email', 'full_name', 'phone', 'created_at']:
        try:
            cur.execute(f"ALTER TABLE users ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()

init_user_table()

# Ensure daily_reports table has all required columns (migration for existing DBs)
def migrate_daily_reports():
    conn = sqlite3.connect("diabetes.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS daily_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            pregnancies REAL,
            glucose REAL,
            bmi REAL,
            blood_pressure REAL,
            skin_thickness REAL,
            insulin REAL,
            dpf REAL,
            age REAL,
            prediction INTEGER,
            probability REAL,
            risk_level TEXT
        )
    """)
    for col in ['pregnancies', 'dpf', 'age']:
        try:
            cur.execute(f"ALTER TABLE daily_reports ADD COLUMN {col} REAL")
        except sqlite3.OperationalError:
            pass  # Column already exists
    conn.commit()
    conn.close()

migrate_daily_reports()

# ---------------- LOAD MODELS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

models = joblib.load(os.path.join(MODEL_DIR, "tri_ensemble.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
imputer = joblib.load(os.path.join(MODEL_DIR, "imputer.pkl"))

rf = models["rf"]
xgb = models["xgb"]
et = models["et"]

# =====================================================
# REGISTER
# =====================================================
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(force=True)

    username = data.get("username")
    password = data.get("password")
    email = data.get("email", "")

    if not username or not password:
        return jsonify({"status": "error", "message": "Missing fields"}), 400

    try:
        conn = sqlite3.connect("diabetes.db")
        cur = conn.cursor()

        cur.execute("INSERT INTO users (username, password, email) VALUES (?,?,?)",
                    (username, password, email))

        conn.commit()
        user_id = cur.lastrowid
        conn.close()

        return jsonify({"status": "success", "user_id": user_id})

    except sqlite3.IntegrityError:
        return jsonify({"status": "error", "message": "User already exists"}), 409

# =====================================================
# LOGIN
# =====================================================
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True)

    username = data.get("username")
    password = data.get("password")

    conn = sqlite3.connect("diabetes.db")
    cur = conn.cursor()

    cur.execute("SELECT id, username, email FROM users WHERE username=? AND password=?",
                (username, password))

    user = cur.fetchone()
    conn.close()

    if user:
        return jsonify({"status": "success", "user_id": user[0], "username": user[1], "email": user[2]})
    else:
        return jsonify({"status": "error", "message": "Invalid credentials"}), 401

# =====================================================
# PROFILE
# =====================================================
@app.route("/profile/<int:user_id>", methods=["GET"])
def get_profile(user_id):
    try:
        conn = sqlite3.connect("diabetes.db")
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT id, username, email, full_name, phone, created_at FROM users WHERE id=?", (user_id,))
        user = cur.fetchone()

        if not user:
            conn.close()
            return jsonify({"status": "error", "message": "User not found"}), 404

        # Get prediction stats
        cur.execute("SELECT COUNT(*) as cnt FROM daily_reports WHERE user_id=?", (user_id,))
        total_predictions = cur.fetchone()["cnt"]

        cur.execute("SELECT COUNT(*) as cnt FROM daily_reports WHERE user_id=? AND prediction=1", (user_id,))
        diabetic_count = cur.fetchone()["cnt"]

        cur.execute("SELECT COUNT(*) as cnt FROM daily_reports WHERE user_id=? AND prediction=0", (user_id,))
        normal_count = cur.fetchone()["cnt"]

        cur.execute("SELECT date FROM daily_reports WHERE user_id=? ORDER BY date DESC LIMIT 1", (user_id,))
        last_row = cur.fetchone()
        last_prediction_date = last_row["date"] if last_row else None

        conn.close()

        return jsonify({
            "profile": {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"] or "",
                "full_name": user["full_name"] or "",
                "phone": user["phone"] or "",
                "created_at": user["created_at"] or ""
            },
            "stats": {
                "total_predictions": total_predictions,
                "diabetic_count": diabetic_count,
                "normal_count": normal_count,
                "last_prediction_date": last_prediction_date
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/profile/<int:user_id>", methods=["PUT"])
def update_profile(user_id):
    data = request.get_json(force=True)

    try:
        conn = sqlite3.connect("diabetes.db")
        cur = conn.cursor()

        cur.execute("SELECT id FROM users WHERE id=?", (user_id,))
        if not cur.fetchone():
            conn.close()
            return jsonify({"status": "error", "message": "User not found"}), 404

        full_name = data.get("full_name", "")
        email = data.get("email", "")
        phone = data.get("phone", "")

        cur.execute("""
            UPDATE users SET full_name=?, email=?, phone=? WHERE id=?
        """, (full_name, email, phone, user_id))

        conn.commit()
        conn.close()

        return jsonify({"status": "success", "message": "Profile updated successfully"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# =====================================================
# PREDICTION
# =====================================================
@app.route("/predict", methods=["POST"])
def predict():
    def safe_float(value):
        try:
            return float(value)
        except:
            return 0.0

    data = request.get_json(force=True)
    logger.info("Prediction request received")

    try:
        # Extract and validate user_id
        user_id = data.get("user_id")
        if not user_id:
            logger.warning("No user_id provided")

        # Extract features in correct order
        pregnancies = safe_float(data.get("Pregnancies"))
        glucose = safe_float(data.get("Glucose"))
        blood_pressure = safe_float(data.get("BloodPressure"))
        skin_thickness = safe_float(data.get("SkinThickness"))
        insulin = safe_float(data.get("Insulin"))
        bmi = safe_float(data.get("BMI"))
        dpf = safe_float(data.get("DiabetesPedigreeFunction"))
        age = safe_float(data.get("Age"))

        # Create feature array (2D array required)
        features = np.array([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]])

        # Apply KNN Imputer
        features = imputer.transform(features)

        # Apply Standard Scaler
        features = scaler.transform(features)

        # Get predictions from all three models
        rf_p = rf.predict_proba(features)[0][1]
        xgb_p = xgb.predict_proba(features)[0][1]
        et_p = et.predict_proba(features)[0][1]

        # Calculate average probability
        avg_probability = float((rf_p + xgb_p + et_p) / 3)
        probability_percentage = round(avg_probability * 100, 2)

        # Determine prediction (0 or 1)
        prediction = 1 if avg_probability >= 0.5 else 0

        # Determine risk level
        if avg_probability < 0.3:
            risk_level = "LOW"
        elif avg_probability < 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        logger.info("Prediction: %s (%s), Risk: %s", prediction, ['Not Diabetic', 'Diabetic'][prediction], risk_level)

        # Save to database if user_id is provided
        if user_id:
            try:
                conn = sqlite3.connect("diabetes.db")
                cur = conn.cursor()

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS daily_reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        pregnancies REAL,
                        glucose REAL,
                        bmi REAL,
                        blood_pressure REAL,
                        skin_thickness REAL,
                        insulin REAL,
                        dpf REAL,
                        age REAL,
                        prediction INTEGER,
                        probability REAL,
                        risk_level TEXT
                    )
                """)

                # Add columns if they don't exist (for existing databases)
                for col in ['pregnancies', 'dpf', 'age']:
                    try:
                        cur.execute(f"ALTER TABLE daily_reports ADD COLUMN {col} REAL")
                    except sqlite3.OperationalError:
                        pass  # Column already exists

                cur.execute("""
                    INSERT INTO daily_reports
                    (user_id, pregnancies, glucose, bmi, blood_pressure, skin_thickness, insulin, dpf, age, prediction, probability, risk_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (user_id, pregnancies, glucose, bmi, blood_pressure, skin_thickness, insulin, dpf, age, prediction, probability_percentage, risk_level))

                conn.commit()
                conn.close()
                logger.info("Saved prediction to database for user %s", user_id)
            except Exception as db_error:
                logger.error("Database save error: %s", db_error)

        # Prepare response
        response = {
            "prediction": prediction,
            "riskLevel": risk_level,
            "probability": probability_percentage,
            "score": round(avg_probability, 3)
        }

        logger.info("Prediction successful: %s", response)

        return jsonify(response)

    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        return jsonify({"status": "error", "message": f"Prediction failed: {str(e)}"}), 500

# =====================================================
# PREDICTION HISTORY
# =====================================================
@app.route("/history/<int:user_id>", methods=["GET"])
def get_history(user_id):
    logger.info("History request for user %s", user_id)

    try:
        conn = sqlite3.connect("diabetes.db")
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("""
            SELECT id, user_id, date, pregnancies, glucose, bmi, blood_pressure,
                   skin_thickness, insulin, dpf, age, prediction, probability, risk_level
            FROM daily_reports
            WHERE user_id = ?
            ORDER BY date DESC
        """, (user_id,))

        rows = cur.fetchall()
        conn.close()

        history = []
        for row in rows:
            history.append({
                "id": row["id"],
                "date": row["date"],
                "pregnancies": row["pregnancies"],
                "glucose": row["glucose"],
                "bmi": row["bmi"],
                "blood_pressure": row["blood_pressure"],
                "skin_thickness": row["skin_thickness"],
                "insulin": row["insulin"],
                "dpf": row["dpf"],
                "age": row["age"],
                "prediction": row["prediction"],
                "probability": row["probability"],
                "risk_level": row["risk_level"]
            })

        logger.info("Found %d history records for user %s", len(history), user_id)

        return jsonify({"history": history})

    except Exception as e:
        logger.error("History error: %s", e, exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

# =====================================================
# MONTHLY REPORT
# =====================================================
@app.route("/monthly-report/<int:user_id>", methods=["GET"])
def monthly_report(user_id):
    month = request.args.get("month")
    year = request.args.get("year")

    logger.info("Monthly report request for user %s: %s/%s", user_id, month, year)

    if not month or not year:
        return jsonify({"status": "error", "message": "Month and year are required"}), 400

    try:
        conn = sqlite3.connect("diabetes.db")
        cur = conn.cursor()

        # Query all records for the user in the specified month/year
        cur.execute("""
            SELECT glucose, bmi, blood_pressure, prediction, probability
            FROM daily_reports
            WHERE user_id = ?
            AND strftime('%m', date) = ?
            AND strftime('%Y', date) = ?
        """, (user_id, month.zfill(2), year))

        records = cur.fetchall()
        conn.close()

        logger.info("Found %d records for monthly report", len(records))

        if not records:
            return jsonify({
                "avg_glucose": 0,
                "avg_bmi": 0,
                "avg_bp": 0,
                "avg_risk": 0,
                "diabetic_days": 0,
                "normal_days": 0,
                "total_records": 0
            })

        # Calculate statistics
        total_glucose = sum(r[0] for r in records)
        total_bmi = sum(r[1] for r in records)
        total_bp = sum(r[2] for r in records)
        total_probability = sum(r[4] for r in records)

        diabetic_days = sum(1 for r in records if r[3] == 1)
        normal_days = len(records) - diabetic_days

        avg_glucose = round(total_glucose / len(records), 2)
        avg_bmi = round(total_bmi / len(records), 2)
        avg_bp = round(total_bp / len(records), 2)
        avg_risk = round(total_probability / len(records), 2)

        response = {
            "avg_glucose": avg_glucose,
            "avg_bmi": avg_bmi,
            "avg_bp": avg_bp,
            "avg_risk": avg_risk,
            "diabetic_days": diabetic_days,
            "normal_days": normal_days,
            "total_records": len(records)
        }

        logger.info("Monthly report generated: glucose=%.2f, bmi=%.2f, bp=%.2f, risk=%.2f%%",
                    avg_glucose, avg_bmi, avg_bp, avg_risk)

        return jsonify(response)

    except Exception as e:
        logger.error("Monthly report error: %s", e, exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

# =====================================================
# SERVE REACT
# =====================================================
@app.route("/")
def serve_react():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def serve_static_files(path):
    file_path = os.path.join(app.static_folder, path)
    if os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")

# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
