import os
import pandas as pd
import io
import csv
from flask import Flask, request, jsonify, render_template, Response
from datetime import datetime

app = Flask(__name__)

# Define the Excel file path for storing candidate inputs
EXCEL_FILE = "candidate_data.xlsx"

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
        return 80.0  # Default if no data exists
    sorted_df = df.sort_values(by="marks", ascending=False)
    count = len(sorted_df)
    top_count = max(1, int(count * 0.001))  # Top 0.1% (or at least one candidate)
    top_candidates = sorted_df.head(top_count)
    return top_candidates["marks"].mean()

def normalize_marks(raw_marks, shift):
    df = load_candidate_data()
    global_mt = compute_top_mean(df)
    session_df = df[df["shift"] == shift]
    session_mt = compute_top_mean(session_df)
    MQ = 30  # Fixed qualifying marks
    if session_mt == MQ:
        normalized = raw_marks
    else:
        normalized = ((global_mt - MQ) / (session_mt - MQ)) * (raw_marks - MQ) + MQ
    return normalized, global_mt, session_mt

def compute_gate_score(marks, mq, mt, sq=350, st=900):
    if marks < mq:
        return 100  # Return a low score for candidates below qualifying marks
    return sq + (st - sq) * ((marks - mq) / (mt - mq))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
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

    branch = "CSE"  # Fixed as CSE

    # Load existing data from the Excel file
    df = load_candidate_data()
    # Treat candidate_id as the primary key: update if exists; otherwise, append.
    if candidate_id in df["candidate_id"].astype(str).values:
        df.loc[df["candidate_id"] == candidate_id, ["marks", "shift", "timestamp"]] = [raw_marks, shift, datetime.utcnow()]
    else:
        new_entry = pd.DataFrame([{
            "candidate_id": candidate_id,
            "marks": raw_marks,
            "branch": branch,
            "shift": shift,
            "timestamp": datetime.utcnow()
        }])
        df = pd.concat([df, new_entry], ignore_index=True)
    save_candidate_data(df)

    normalized_marks, global_mt, session_mt = normalize_marks(raw_marks, shift)
    gate_score = compute_gate_score(normalized_marks, 30, global_mt)
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

# Admin route: Returns all candidate data as JSON.
@app.route("/admin/data", methods=["GET"])
def admin_data():
    df = load_candidate_data()
    data = df.to_dict(orient="records")
    return jsonify(data)

# Admin route: Download candidate data as CSV.
@app.route("/admin/download", methods=["GET"])
def admin_download():
    df = load_candidate_data()
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(["Candidate ID", "Marks", "Branch", "Shift", "Timestamp"])
    for _, row in df.iterrows():
        cw.writerow([row["candidate_id"], row["marks"], row["branch"], row["shift"], row["timestamp"]])
    output = si.getvalue()
    return Response(output, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment;filename=candidate_data.csv"})

if __name__ == "__main__":
    # In production, run using a WSGI server (e.g., gunicorn via a Procfile)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
