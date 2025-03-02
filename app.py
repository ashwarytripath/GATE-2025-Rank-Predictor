import os
import re
import pandas as pd
import io
import csv
from flask import Flask, request, jsonify, render_template, Response
from datetime import datetime

app = Flask(__name__)

EXCEL_FILE = "candidate_data.xlsx"

# ------------------------------------------------------------------------------
# 1) DATA CLEANING
# ------------------------------------------------------------------------------
def clean_candidate_data(df):
    """
    Clean and validate candidate data:
      - candidate_id must match the pattern CS##S######## (uppercase).
      - marks must be numeric and between 0 and 100.
      - shift must be "Morning" or "Afternoon".
      - branch must be "CSE".
    Invalid rows are removed.
    """
    if df.empty:
        return df

    # Normalize candidate_id
    df["candidate_id"] = df["candidate_id"].astype(str).str.strip().str.upper()
    pattern = r'^CS\d{2}S\d{8}$'
    df = df[df["candidate_id"].str.fullmatch(pattern, na=False)]

    # Normalize marks
    df["marks"] = pd.to_numeric(df["marks"], errors="coerce")
    df = df[df["marks"].between(0, 100)]

    # Normalize shift
    df["shift"] = df["shift"].astype(str).str.strip().str.capitalize()
    df = df[df["shift"].isin(["Morning", "Afternoon"])]

    # Normalize branch (should be "CSE")
    df["branch"] = df["branch"].astype(str).str.strip().str.upper()
    df = df[df["branch"] == "CSE"]

    return df

# ------------------------------------------------------------------------------
# 2) FILE LOAD/SAVE
# ------------------------------------------------------------------------------
def load_candidate_data():
    if os.path.exists(EXCEL_FILE):
        try:
            df = pd.read_excel(EXCEL_FILE)
            df = clean_candidate_data(df)
            # Optionally save cleaned data back to Excel
            save_candidate_data(df)
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

# ------------------------------------------------------------------------------
# 3) HELPER FUNCTIONS FOR GATE CALCULATIONS
# ------------------------------------------------------------------------------
def compute_cutoff(df):
    """
    Compute M_q for general category using the GATE formula:
       M_q = max(25, min(40, mu + sigma))
    where mu is the mean, sigma is the std dev of all marks.
    """
    if df.empty:
        return 25.0  # Fallback if no data

    mu = df["marks"].mean()
    sigma = df["marks"].std(ddof=1)  # sample std dev
    if pd.isna(sigma):
        sigma = 0.0
    return max(25, min(40, mu + sigma))

def compute_cutoff_for_session(df, shift):
    """
    Compute M_q for the session (e.g. "Morning" or "Afternoon")
    using the same formula. This is needed for multi-session normalization.
    """
    session_df = df[df["shift"] == shift]
    if session_df.empty:
        return 25.0
    mu_s = session_df["marks"].mean()
    sigma_s = session_df["marks"].std(ddof=1)
    if pd.isna(sigma_s):
        sigma_s = 0.0
    return max(25, min(40, mu_s + sigma_s))

def compute_top_mean(df):
    """
    Compute M_t = mean of top 0.1% marks (or at least 1 candidate).
    Could also enforce a minimum of top 10 if desired.
    """
    if df.empty:
        return 80.0
    sorted_df = df.sort_values(by="marks", ascending=False)
    count = len(sorted_df)
    top_count = max(1, int(count * 0.001))
    top_candidates = sorted_df.head(top_count)
    return top_candidates["marks"].mean()

def normalize_marks(raw_marks, shift):
    """
    Multi-session normalization (approx GATE approach):
       M_ij = M_q_global + ( (M_t_global - M_q_global)/(M_t_session - M_q_session) ) * ( raw_marks - M_q_session )
    If session top mean == session M_q => fallback to raw_marks.
    """
    df = load_candidate_data()

    # Global cutoff & top mean
    M_q_global = compute_cutoff(df)
    M_t_global = compute_top_mean(df)

    # Session cutoff & top mean
    M_q_session = compute_cutoff_for_session(df, shift)
    session_df = df[df["shift"] == shift]
    M_t_session = compute_top_mean(session_df)

    # If M_t_session == M_q_session => fallback
    if M_t_session == M_q_session:
        normalized = raw_marks
    else:
        normalized = M_q_global + ((M_t_global - M_q_global) / (M_t_session - M_q_session)) * (raw_marks - M_q_session)

    return normalized, M_t_global, M_t_session, M_q_global

def compute_gate_score(marks, df):
    """
    1) Compute M_q (cutoff) from entire data.
    2) Compute M_t (top mean).
    3) If marks < M_q => 100.
    4) Else linear interpolation between S_q=350 and S_t=1000.
    """
    M_q = compute_cutoff(df)      # cutoff for entire data
    M_t = compute_top_mean(df)    # top mean for entire data
    S_q = 350
    S_t = 1000

    if marks < M_q:
        return 100.0

    return S_q + (S_t - S_q) * ((marks - M_q) / (M_t - M_q))

# ------------------------------------------------------------------------------
# 4) ROUTES
# ------------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Normalize candidate_id
    candidate_id = data.get("candidate_id", "").strip().upper()
    raw_marks = data.get("rawMarks")
    shift = data.get("shift")

    # Validate required fields
    if not candidate_id or raw_marks is None or shift is None:
        return jsonify({"error": "Candidate ID, rawMarks, and shift are required"}), 400

    # Validate candidate_id format
    pattern = r'^CS\d{2}S\d{8}$'
    if not re.fullmatch(pattern, candidate_id):
        return jsonify({"error": "Candidate ID must be in format CS##S######## (e.g., CS25S13049105)"}), 400

    # Validate marks
    try:
        raw_marks = float(raw_marks)
        if raw_marks < 0 or raw_marks > 100:
            return jsonify({"error": "Marks must be between 0 and 100."}), 400
    except ValueError:
        return jsonify({"error": "rawMarks must be a number"}), 400

    # Load existing data & update or append
    df = load_candidate_data()
    branch = "CSE"

    if candidate_id in df["candidate_id"].astype(str).values:
        df.loc[df["candidate_id"] == candidate_id, ["marks", "shift", "timestamp"]] = [
            raw_marks, shift, datetime.utcnow()
        ]
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

    # Multi-session normalization
    normalized_marks, global_mt, session_mt, M_q_global = normalize_marks(raw_marks, shift)

    # Final GATE score
    gate_score = compute_gate_score(normalized_marks, df)

    user_count = df["candidate_id"].nunique()

    return jsonify({
        "candidate_id": candidate_id,
        "rawMarks": raw_marks,
        "normalizedMarks": round(normalized_marks, 2),
        "gateScore": round(gate_score, 2),
        "globalMt": round(global_mt, 2),
        "sessionMt": round(session_mt, 2),
        "cutoffGlobal": round(M_q_global, 2),
        "userCount": user_count
    })

@app.route("/admin/data", methods=["GET"])
def admin_data():
    df = load_candidate_data()
    data = df.to_dict(orient="records")
    return jsonify(data)

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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
