# Yellow.ai Q&A Uploader


Upload a sheet with a `question` column; get back an `answer` column via the Yellow.ai (Azure OpenAI-compatible) chat completions API.


## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env to add your key
python app.py