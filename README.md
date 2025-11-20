# 🟡 Yellow.ai RFP Compliance Matrix Automation Tool

A lightweight **Flask web app** to automate generating RFP (Request for Proposal) compliance matrix answers using **Yellow.ai’s Knowledge Base RAG API** (also compatible with **Azure OpenAI** format).

Simply upload a `.csv` or `.xlsx` file with a `question` column, and you’ll receive the same sheet with an added `answer` column — generated intelligently via the Yellow.ai API or any other RAG API.

---

## 🚀 Features

- ✅ Upload `.csv` or `.xlsx` RFP compliance sheets  
- ✅ Auto-generate AI answers for every question  
- ✅ Skip empty rows in the `question` column automatically  
- ✅ Download processed sheet instantly  
- ✅ Built-in logging (see logs in the terminal)  
- ✅ Persistent — reupload as many times as needed without restarting the app  

---

## 🧩 Requirements

- Python 3.9+
- Flask
- Pandas
- Requests
- (All dependencies listed in `requirements.txt`)

---

## ⚙️ Setup & Run Locally

```bash
# 1️⃣ Create and activate virtual environment
python -m venv .venv
# On macOS/Linux
source .venv/bin/activate
# On Windows PowerShell
.\.venv\Scripts\Activate.ps1

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Set up environment variables
cp .env.example .env
# Open .env and add your Yellow.ai API key

# 4️⃣ Run the app
python app.py
