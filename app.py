import streamlit as st
import pandas as pd
from detective import investigate
from agent import decide, ask_followup
from nlp_tools import top_keywords, sentiment_summary

st.set_page_config(page_title="AI Data Detective", page_icon="🕵️", layout="wide")
st.title("🕵️ AI Data Detective & Decision Agent")

file = st.sidebar.file_uploader("Upload CSV", type="csv")
goal = st.sidebar.text_input("Business goal (optional)", placeholder="e.g. reduce churn")

if file:
    df = pd.read_csv(file)
    st.subheader("Data preview")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("🔍 Investigate"):
        with st.spinner("Collecting evidence..."):
            ev = investigate(df)

            # Text analysis: only for columns that look like free text
            for col in ev["profile"]["text_cols"][:2]:
                if df[col].astype(str).str.len().mean() > 30:
                    ev[f"text_{col}"] = {
                        "keywords": top_keywords(df[col]),
                        "sentiment": sentiment_summary(df[col]),
                    }
            st.session_state.evidence = ev

        with st.spinner("Agent reasoning..."):
            st.session_state.report = decide(st.session_state.evidence, goal)

    if "report" in st.session_state:
        r, ev = st.session_state.report, st.session_state.evidence

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", ev["profile"]["rows"])
        c2.metric("Duplicates", ev["profile"]["duplicate_rows"])
        c3.metric("Anomalies", ev["anomaly_count"])

        st.header("Summary")
        st.write(r["summary"])

        st.header("Key findings")
        for f in r["key_findings"]:
            st.markdown(f"- {f}")

        st.header("Decisions")
        for d in r["decisions"]:
            st.info(f"**[{d['priority']}]** {d['action']}\n\n{d['reason']}  (confidence: {d['confidence']}%)")

        with st.expander("Raw evidence"):
            st.json(ev)

        st.header("Ask the detective")
        q = st.text_input("Follow-up question")
        if q:
            st.write(ask_followup(ev, q))
else:
    st.info("Upload a CSV in the sidebar to begin.")