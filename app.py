import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from datetime import datetime

app = Flask(__name__)

# Define the Excel file path for storing candidate inputs
EXCEL_FILE = "gate_data.xlsx"

def load_candidate_data():
    if os.path.exists(EXCEL_FILE):
        try:
            df = pd.read_excel(EXCEL_FILE)
            return df
        except Exception as e:
            app.logger.error(f"Error reading Excel file: {e}")
            return pd.DataFrame(columns=["candidate_id", "marks", "branch", "shift", "timestamp"])
    else:
        return pd.DataFrame(columns=["candidate_id", "marks", "branch", "shift", "timestamp"])

def save_candidate_data(df):
    try:
        df.to_excel(EXCEL_FILE, index=False)
    except Exception as e:
        app.logger.error(f"Error writing Excel file: {e}")

def compute_top_mean(df):
    if df.empty:
        return 80.0  # Default value if no data exists
    sorted_df = df.sort_values(by="marks", ascending=False)
    count = len(sorted_df)
    top_count = max(1, int(count * 0.001))  # Top 0.1% (or at least one candidate)
    top_candidates = sorted_df.head(top_count)
    return top_candidates["marks"].mean()

def normalize_marks(raw_marks, user_shift):
    df = load_candidate_data()
    global_mt = compute_top_mean(df)
    session_df = df[df["shift"] == user_shift]
    session_mt = compute_top_mean(session_df)
    MQ = 30  # Fixed qualifying marks
    if session_mt == MQ:
        normalized = raw_marks
    else:
        normalized = ((global_mt - MQ) / (session_mt - MQ)) * (raw_marks - MQ) + MQ
    return normalized, global_mt, session_mt

def compute_gate_score(marks, mq, mt, sq=350, st=900):
    if marks < mq:
        return 100  # Low score for candidates below qualifying marks
    return sq + (st - sq) * ((marks - mq) / (mt - mq))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        candidate_id = data.get("candidate_id")
        raw_marks = data.get("rawMarks")
        shift = data.get("shift")
        if candidate_id is None or raw_marks is None or shift is None:
            return jsonify({"error": "Candidate ID, rawMarks, and shift are required"}), 400

        try:
            raw_marks = float(raw_marks)
        except ValueError:
            return jsonify({"error": "rawMarks must be a number"}), 400

        branch = "CSE"  # Hardcoded as CSE

        # Load current candidate data
        df = load_candidate_data()
        # Check if the candidate_id already exists (treat candidate_id as primary key)
        if candidate_id in df["candidate_id"].astype(str).values:
            # Update existing record
            df.loc[df["candidate_id"] == candidate_id, ["marks", "shift", "timestamp"]] = [raw_marks, shift, datetime.utcnow()]
        else:
            # Append new record
            new_entry = pd.DataFrame([{
                "candidate_id": candidate_id,
                "marks": raw_marks,
                "branch": branch,
                "shift": shift,
                "timestamp": datetime.utcnow()
            }])
            df = pd.concat([df, new_entry], ignore_index=True)
        save_candidate_data(df)

        # Compute normalization and GATE Score
        normalized_marks, global_mt, session_mt = normalize_marks(raw_marks, shift)
        gate_score = compute_gate_score(normalized_marks, 30, global_mt)

        # Count unique users based on candidate_id
        user_count = df["candidate_id"].nunique()

        return jsonify({
            "candidate_id": candidate_id,
            "rawMarks": raw_marks,
            "normalizedMarks": round(normalized_marks, 2),
            "gateScore": round(gate_score, 2),
            "globalMt": round(global_mt, 2),
            "sessionMt": round(session_mt, 2),
            "userCount": user_count
        })
    except Exception as e:
        app.logger.error(f"Error in /api/predict: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # For production deployment, use a WSGI server (e.g., gunicorn via a Procfile)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
