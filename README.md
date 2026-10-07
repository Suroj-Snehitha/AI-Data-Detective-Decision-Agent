# 🕵️ AI Data Detective & Decision Agent

A Streamlit app that investigates any CSV file and tells you what to do about it.

Upload a dataset and the app will:

1. **Investigate** the data: missing values, duplicates, outliers, anomalies, correlations, and text sentiment.
2. **Explain** the findings in plain English using Google Gemini.
3. **Decide**: produce prioritized, actionable recommendations with confidence scores.
4. **Answer follow-up questions** about your data.

## How it works

```
CSV upload -> Profiler -> Detective (ML anomalies, correlations, NLP)
           -> Evidence (JSON) -> LLM Agent (Gemini) -> Decisions -> Streamlit UI
```

**Core design idea:** Python computes the facts, and the LLM only interprets them. The model never sees your full dataset, only a compact summary of evidence. This avoids arithmetic mistakes and made-up numbers, keeps token costs low, and limits how much raw data leaves your machine.

## Features

| Feature | Technique |
|---|---|
| Data quality profile | pandas (missing %, duplicates, dtypes) |
| Column-level outliers | IQR rule |
| Row-level anomalies | scikit-learn Isolation Forest |
| Strong correlations | Pearson correlation threshold |
| Text keywords and entities | NLTK, spaCy |
| Sentiment analysis | Hugging Face DistilBERT (runs locally on CPU) |
| Reasoning and decisions | Google Gemini |
| Interactive UI | Streamlit |

## Project structure

```
ai-data-detective/
├── app.py            # Streamlit UI
├── detective.py      # Data profiling and ML analysis
├── nlp_tools.py      # Keywords, entities, sentiment
├── agent.py          # Gemini reasoning and decisions
├── requirements.txt
├── .env              # Your API key (never commit this)
└── README.md
```

## Setup

### 1. Clone or create the project folder

```bash
cd ai-data-detective
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

> `torch` and `transformers` are large (several GB). The install can take a while.

### 4. Add your Gemini API key

Get a free key from [Google AI Studio](https://aistudio.google.com), then create a file named `.env` in the project root:

```
GEMINI_API_KEY=your_key_here
```

Write it exactly like this: no quotes and no spaces around `=`.

### 5. Run the app

```bash
streamlit run app.py
```

Open the URL shown in your terminal (usually http://localhost:8501).

## Usage

1. Upload a CSV from the sidebar.
2. (Optional) Enter a business goal, such as "reduce customer churn".
3. Click **Investigate**.
4. Read the summary, key findings, and decisions.
5. Expand **Raw evidence** to see exactly what was sent to the LLM.
6. Ask follow-up questions in the **Ask the detective** box.

Good test datasets: Titanic, Telco Customer Churn, or any CSV from Kaggle.

## Configuration

| Setting | Where | Notes |
|---|---|---|
| `GEMINI_API_KEY` | `.env` | Required |
| Gemini model name | `agent.py` | Defaults to `gemini-flash-latest`, an alias that always points to the current Flash model |
| Anomaly sensitivity | `detective.py` (`contamination`) | Default `0.05` (flags the top 5%) |
| Correlation threshold | `detective.py` (`threshold`) | Default `0.7` |
| Evidence size sent to LLM | `agent.py` | Truncated to 12,000 characters |

## Troubleshooting

**`404 This model ... is no longer available`**
Google retires old Gemini models. In `agent.py`, use `genai.GenerativeModel("gemini-flash-latest")`, or run this to see which models your key can use:

```bash
python -c "import os; from dotenv import load_dotenv; import google.generativeai as genai; load_dotenv(); genai.configure(api_key=os.getenv('GEMINI_API_KEY')); [print(m.name) for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]"
```

**`NameError: name 'df' is not defined`**
All code that uses `df` must be indented inside the `if file:` block in `app.py`.

**`429` error**
You hit the free-tier rate limit. Wait a minute and try again.

**`LookupError: stopwords not found`**
Run `python -c "import nltk; nltk.download('stopwords')"`.

**spaCy model not found**
Run `python -m spacy download en_core_web_sm`.

**First sentiment run is slow**
The DistilBERT model (about 250 MB) downloads once, then it is cached.

**Large CSV is slow or crashes**
Sample it before analysis, for example `df = df.sample(50000)`.

**`.env` changes are ignored**
Restart Streamlit. The `.env` file is only read at startup.

## Privacy

- The full dataset is never sent to Gemini, only aggregated evidence.
- The evidence includes up to 5 anomalous rows, which contain real values from your data.
- Do not upload sensitive or personal data unless you mask those rows first.
- Add `.env` and `venv/` to `.gitignore` so your API key is never committed.

## Roadmap

- [ ] Charts for correlations and anomalies
- [ ] Time-series trend-break detection
- [ ] Function-calling agent (Gemini chooses which analysis tools to run)
- [ ] LangChain memory for follow-up conversations
- [ ] Export report as PDF or Markdown
- [ ] Caching with `@st.cache_data`
- [ ] Toggle to mask sensitive values before sending evidence to the LLM

## Tech stack

Python, Streamlit, pandas, NumPy, scikit-learn, Google Gemini (`google-generativeai`), NLTK, spaCy, Hugging Face Transformers, PyTorch, LangChain.

## License

MIT. Add a `LICENSE` file if you publish the project.