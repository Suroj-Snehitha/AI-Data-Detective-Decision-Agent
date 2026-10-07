import spacy
from collections import Counter
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")
STOP = set(stopwords.words("english"))
_sentiment = None


def top_keywords(texts, n=10):
    words = []
    for t in texts.dropna().astype(str).head(500):
        words += [w.lower() for w in t.split() if w.isalpha() and w.lower() not in STOP]
    return Counter(words).most_common(n)


def top_entities(texts, n=10):
    ents = Counter()
    for doc in nlp.pipe(texts.dropna().astype(str).head(200)):
        for e in doc.ents:
            ents[(e.text, e.label_)] += 1
    return ents.most_common(n)


def sentiment_summary(texts):
    """Lazy-load a Hugging Face model so the app starts fast."""
    global _sentiment
    if _sentiment is None:
        from transformers import pipeline
        _sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
    sample = texts.dropna().astype(str).head(100).str[:512].tolist()
    results = _sentiment(sample)
    pos = sum(r["label"] == "POSITIVE" for r in results)
    return {"positive_pct": round(100 * pos / len(results), 1), "sampled": len(results)}