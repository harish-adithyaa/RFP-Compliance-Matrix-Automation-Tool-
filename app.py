import os
import io
import time
import uuid
import logging
from pathlib import Path
from typing import Tuple

from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename
import pandas as pd
import requests
from dotenv import load_dotenv

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

# ---- Config ----
ALLOWED_EXTS = {".csv", ".xlsx", ".xls"}
TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)

YELLOW_AI_API_URL = os.getenv(
    "YELLOW_AI_API_URL",
    ""
).strip()
YELLOW_AI_API_KEY = os.getenv("YELLOW_AI_API_KEY", "").strip()
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a Yellow.ai assistant that responds to user queries related to some functional and techincal requirments, strictly based on the information available on docs.yellow.ai website. Keep responses accurate, concise, clear, and under 250 words."
)

# Logging config via env (optional)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()   # DEBUG, INFO, WARNING, ERROR
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "false").lower() in {"1", "true", "yes", "y"}

# ----------------------------
# Logging setup
# ----------------------------
log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
if LOG_TO_FILE:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format=log_format,
        filename="app.log",
        filemode="a",
    )
else:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format=log_format,
    )

logger = logging.getLogger("qa-uploader")

# Sanity checks for required env
if not YELLOW_AI_API_URL:
    logger.error("YELLOW_AI_API_URL is not set. See .env.example")
    raise SystemExit("YELLOW_AI_API_URL is not set. See .env.example")
if not YELLOW_AI_API_KEY:
    logger.error("YELLOW_AI_API_KEY is not set. See .env.example")
    raise SystemExit("YELLOW_AI_API_KEY is not set. See .env.example")

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", str(uuid.uuid4()))  # for flash() messages


# ----------------------------
# Helpers
# ----------------------------
def is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTS


def read_dataframe(file_storage) -> Tuple[pd.DataFrame, str]:
    """
    Return (df, source_ext). Raises ValueError on invalid input.
    """
    filename = secure_filename(file_storage.filename or "")
    ext = Path(filename).suffix.lower()
    logger.info("Attempting to read upload: name=%s ext=%s", filename, ext)

    if ext not in ALLOWED_EXTS:
        raise ValueError("Unsupported file type. Please upload .csv, .xlsx or .xls")

    try:
        if ext == ".csv":
            df = pd.read_csv(file_storage.stream)
        elif ext == ".xlsx":
            df = pd.read_excel(file_storage.stream, engine="openpyxl")
        else:  # .xls
            df = pd.read_excel(file_storage.stream, engine="xlrd")
    except Exception as e:
        logger.exception("Failed reading the uploaded file")
        raise ValueError(f"Could not read the file. Make sure it's a valid {ext} – {e}")

    if df.empty:
        raise ValueError("The uploaded file has no rows.")

    # Find 'question' column case-insensitively
    lower_map = {c.lower(): c for c in df.columns}
    if "question" not in lower_map:
        if "questions" in lower_map:
            original = lower_map["questions"]
            df.rename(columns={original: "question"}, inplace=True)
            logger.info("Renamed column '%s' -> 'question'", original)
        else:
            raise ValueError("Missing required column 'question'. Add a header named 'question'.")
    else:
        original = lower_map["question"]
        if original != "question":
            df.rename(columns={original: "question"}, inplace=True)
            logger.info("Normalized column '%s' -> 'question'", original)

    # Normalize to strings for safety (don't force NaN → 'nan')
    return df, ext


def redact(text: str, keep_last: int = 4) -> str:
    """Redact sensitive values (e.g., API key) for logging."""
    if not text:
        return ""
    if len(text) <= keep_last:
        return "*" * len(text)
    return "*" * (len(text) - keep_last) + text[-keep_last:]


def call_yellow_ai(question: str, session: requests.Session, max_retries: int = 3, timeout: int = 30) -> str:
    """
    Call the Yellow.ai (Azure OpenAI-compatible) chat completions endpoint and return the answer text.
    """
    if not isinstance(question, str) or not question.strip():
        # Explicitly skip empties here as well
        return ""

    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question.strip()},
        ]
    }
    headers = {
        "Content-Type": "application/json",
        # Some gateways are case-sensitive; send both variants safely
        "api-key": YELLOW_AI_API_KEY,
        "Api-Key": YELLOW_AI_API_KEY,
    }

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("API request (attempt %d): %s", attempt, question.strip())
            resp = session.post(YELLOW_AI_API_URL, headers=headers, json=payload, timeout=timeout)

            if resp.status_code == 429:
                delay = min(2 ** attempt, 10)
                logger.warning("Rate limited (429). Backing off for %ss and retrying...", delay)
                time.sleep(delay)
                continue

            resp.raise_for_status()
            # Log a trimmed response for visibility (avoid logging huge text)
            trimmed = (resp.text[:500] + "…") if len(resp.text) > 500 else resp.text
            logger.debug("API raw response: %s", trimmed)

            data = resp.json()

            # Typical Azure/Yellow output shape
            choice0 = (data.get("choices") or [{}])[0]
            msg = choice0.get("message") or {}
            content = msg.get("content")
            if content:
                answer = str(content).strip()
                logger.info("Parsed answer (%d chars)", len(answer))
                return answer

            # Fallbacks (defensive): some providers return 'text'
            content = choice0.get("text")
            if content:
                answer = str(content).strip()
                logger.info("Parsed answer from 'text' (%d chars)", len(answer))
                return answer

            logger.error("No parsable content in API response for question: %s", question.strip())
            return ""
        except requests.RequestException as e:
            logger.error("API error on attempt %d: %s", attempt, e)
            if attempt >= max_retries:
                return f"[Error contacting API: {e}]"
            time.sleep(min(2 ** attempt, 10))
    return ""


def write_output(df: pd.DataFrame, source_ext: str) -> Tuple[bytes, str, str]:
    """
    Return (bytes_data, download_filename, mimetype).
    We preserve CSV; Excel becomes .xlsx for reliability.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = f"answers-{stamp}"

    if source_ext == ".csv":
        out_name = f"{base}.csv"
        data = df.to_csv(index=False).encode("utf-8")
        return data, out_name, "text/csv"
    else:
        out_name = f"{base}.xlsx"
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        buf.seek(0)
        return buf.read(), out_name, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def index():
    return render_template("index.html")


@app.post("/process")
def process():
    file = request.files.get("file")
    if not file or not file.filename:
        flash("Please choose a file to upload.", "error")
        return redirect(url_for("index"))

    if not is_allowed(file.filename):
        flash("Unsupported file type. Upload a .csv, .xlsx or .xls.", "error")
        return redirect(url_for("index"))

    logger.info("Upload received: %s", secure_filename(file.filename))

    try:
        df, ext = read_dataframe(file)
    except ValueError as e:
        logger.warning("Validation failed: %s", e)
        flash(str(e), "error")
        return redirect(url_for("index"))

    # Ensure an 'answer' column exists (we'll overwrite/populate it)
    if "answer" not in df.columns:
        df.insert(len(df.columns), "answer", "")

    session = requests.Session()

    # Iterate rows and call API (skip empty questions)
    processed, skipped = 0, 0
    for idx, row in df.iterrows():
        raw_q = row.get("question", "")
        # Robust empty check: NaN / None / whitespace only
        is_empty = False
        if raw_q is None:
            is_empty = True
        else:
            if isinstance(raw_q, float) and pd.isna(raw_q):
                is_empty = True
            elif isinstance(raw_q, str) and not raw_q.strip():
                is_empty = True

        if is_empty:
            df.at[idx, "answer"] = ""
            skipped += 1
            logger.info("Row %s: empty question -> skipped", idx)
            continue

        q = str(raw_q).strip()
        logger.info("Row %s: sending question (%d chars)", idx, len(q))
        answer = call_yellow_ai(q, session)
        df.at[idx, "answer"] = answer
        processed += 1

    logger.info("Completed processing. Rows processed=%d, skipped=%d", processed, skipped)

    # Return file for download via a temporary tokenized URL
    data, filename, mimetype = write_output(df, ext)
    token = str(uuid.uuid4())
    out_path = TMP_DIR / f"{token}-{filename}"
    with open(out_path, "wb") as f:
        f.write(data)

    logger.info("Output ready: %s (mime=%s)", out_path.name, mimetype)
    return render_template("result.html", download_url=url_for("download", token=token, filename=filename))


@app.get("/download/<token>/<path:filename>")
def download(token: str, filename: str):
    path = TMP_DIR / f"{token}-{filename}"
    if not path.exists():
        logger.warning("Download attempted for missing/expired path: %s", path)
        flash("Your file link has expired or is invalid. Please re-upload.", "error")
        return redirect(url_for("index"))
    logger.info("Download served: %s", path.name)
    return send_file(path, as_attachment=True, download_name=filename, mimetype=None)


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", "5000"))
    redacted_key = redact(YELLOW_AI_API_KEY)
    logger.info("Starting app on port %d | API_URL=%s | API_KEY=%s | LOG_TO_FILE=%s",
                port, YELLOW_AI_API_URL, redacted_key, LOG_TO_FILE)
    app.run(host="127.0.0.1", port=port, debug=True)
