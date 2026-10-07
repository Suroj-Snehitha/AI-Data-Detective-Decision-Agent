import os, json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Model names change over time; check ai.google.dev for the current ones.
model = genai.GenerativeModel("gemini-flash-latest")

SYSTEM = """You are a senior data analyst.
You receive EVIDENCE computed by Python. Use ONLY that evidence; never invent numbers.
Return strictly valid JSON:
{
 "summary": "3-4 sentence plain-English overview",
 "key_findings": ["..."],
 "decisions": [
   {"action": "...", "reason": "...", "priority": "High|Medium|Low", "confidence": 0-100}
 ],
 "risks": ["..."]
}"""


def decide(evidence: dict, business_goal: str = "") -> dict:
    prompt = f"{SYSTEM}\n\nBUSINESS GOAL: {business_goal or 'General data health check'}\n\nEVIDENCE:\n{json.dumps(evidence, default=str)[:12000]}"
    resp = model.generate_content(prompt)
    text = resp.text.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"summary": text, "key_findings": [], "decisions": [], "risks": []}


def ask_followup(evidence: dict, question: str) -> str:
    prompt = f"Using only this evidence, answer concisely:\n{json.dumps(evidence, default=str)[:12000]}\n\nQ: {question}"
    return model.generate_content(prompt).text