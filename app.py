
import time
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

st.set_page_config(page_title="Global Personal Finance Advisor", page_icon="💰", layout="wide")
st.title("💰 Global Personal Finance Advisor")
st.caption("RAG-based financial education and illustrative planning tools")

COUNTRIES = {
    "All (exploratory)": "ALL", "Global-only": "GLOBAL",
    "United States": "US", "India": "IN",
    "United Kingdom": "GB", "Canada": "CA", "Australia": "AU"
}
country = COUNTRIES[st.sidebar.selectbox("Jurisdiction", list(COUNTRIES))]
currency = st.sidebar.text_input("Currency code", "USD").upper()
model = st.sidebar.text_input("Gemini model", "gemini-2.5-flash")
page = st.sidebar.radio("Modules", [
    "Finance Chat", "Budget", "Investment", "Loans",
    "Tax", "Insurance", "Retirement", "Financial Goals"
])
api_key = st.secrets.get("GEMINI_API_KEY", "")

@st.cache_resource
def load_resources():
    base = Path(__file__).parent
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(base / "chroma_db"))
    collection = client.get_collection("global_finance_knowledge")
    return embedder, client, collection

try:
    embedder, chroma_client, collection = load_resources()
except Exception as exc:
    st.error("Knowledge base could not be loaded. Include the complete chroma_db folder.")
    st.exception(exc)
    st.stop()

def retrieve(query, category=None, k=5):
    total = collection.count()
    if total == 0:
        return []
    filters = []
    if country == "GLOBAL":
        filters.append({"country": "GLOBAL"})
    elif country != "ALL":
        filters.append({"$or": [{"country": country}, {"country": "GLOBAL"}]})
    if category:
        filters.append({"category": category})
    where = filters[0] if len(filters) == 1 else ({"$and": filters} if len(filters) > 1 else None)
    vector = embedder.encode([query], normalize_embeddings=True).tolist()
    args = {"query_embeddings": vector, "n_results": min(k, total),
            "include": ["documents", "metadatas", "distances"]}
    if where is not None:
        args["where"] = where
    result = collection.query(**args)
    return [{"text": d, **m, "distance": float(dist)}
            for d, m, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])]

def call_gemini(prompt):
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured in Streamlit Secrets.")
    last_error = None
    for attempt in range(3):
        try:
            with genai.Client(api_key=api_key) as client:
                response = client.models.generate_content(
                    model=model, contents=prompt, config={"temperature": 0.2})
            return response.text or "Gemini returned an empty response."
        except Exception as exc:
            last_error = exc
            message = str(exc).upper()
            transient = any(x in message for x in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"])
            if not transient or attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))
    raise last_error

def rag(query, category=None):
    sensitive = ["tax rate", "tax deduction", "tax filing", "tax bracket", "tax law", "tax liability", "regulation"]
    if country in ("ALL", "GLOBAL") and any(x in query.lower() for x in sensitive):
        return "Select a specific country for jurisdiction-specific tax or regulatory information.", []
    docs = retrieve(query, category)
    if not docs:
        return "No relevant evidence found. Check uploaded PDFs and metadata.", []
    if not api_key:
        return "App administrator: add GEMINI_API_KEY under Streamlit App settings → Secrets.", []
    context = "\n\n".join([
        f"Source: {d.get('source')} | Publisher: {d.get('publisher')} | Country: {d.get('country')} | Page: {d.get('page')}\n{d['text']}"
        for d in docs])
    prompt = f"""
You are an educational global finance assistant.
Use retrieved documents as primary evidence. Do not mix jurisdictions or invent laws,
tax rates, fees, returns or guarantees. Disclose missing evidence. Do not give
personalized buy/sell recommendations. Cite filenames and page numbers.

Jurisdiction: {country}
Question: {query}
Retrieved context:
{context}
"""
    try:
        answer = call_gemini(prompt)
    except Exception as exc:
        msg = str(exc).upper()
        if "503" in msg or "UNAVAILABLE" in msg:
            answer = "Gemini is temporarily overloaded. Please retry later."
        elif "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            answer = "Gemini API quota/rate limit reached. Check your API quota."
        else:
            answer = f"Gemini API error: {type(exc).__name__}: {exc}"
    refs = [{k: d.get(k) for k in ["source", "publisher", "country", "page", "url"]} for d in docs]
    return answer, refs

def emi_calc(p, annual_rate, months):
    if p <= 0 or months <= 0 or annual_rate < 0:
        raise ValueError("Invalid loan inputs.")
    r = annual_rate / 1200
    emi = p / months if r == 0 else p * r * (1 + r) ** months / ((1 + r) ** months - 1)
    total = emi * months
    return emi, total, total - p

def fv(monthly, annual_rate, years):
    n = int(years * 12)
    r = annual_rate / 1200
    if monthly < 0 or annual_rate < 0 or years < 0:
        raise ValueError("Inputs cannot be negative.")
    if n == 0:
        return 0.0
    return monthly * n if r == 0 else monthly * (((1 + r) ** n - 1) / r) * (1 + r)

if page == "Finance Chat":
    st.header("AI Finance Chat")
    q = st.chat_input("Ask a financial education question")
    if q:
        with st.spinner("Retrieving and generating..."):
            ans, refs = rag(q)
        st.markdown(ans)
        if refs:
            st.subheader("Retrieved sources")
            st.dataframe(pd.DataFrame(refs), use_container_width=True)

elif page == "Budget":
    st.header("Monthly Budget Planner")
    income = st.number_input("Monthly income", min_value=0.0, value=4000.0)
    cats = ["Housing", "Food", "Transport", "Utilities", "Healthcare", "Education", "Shopping", "Other"]
    expenses = {c: st.number_input(c, min_value=0.0, value=0.0, key="budget_" + c) for c in cats}
    total = sum(expenses.values())
    balance = income - total
    a, b, c = st.columns(3)
    a.metric("Income", f"{currency} {income:,.2f}")
    b.metric("Expenses", f"{currency} {total:,.2f}")
    c.metric("Remaining", f"{currency} {balance:,.2f}")
    st.metric("Remaining %", f"{balance / income * 100 if income else 0:.2f}%")
    if total:
        st.plotly_chart(px.pie(names=list(expenses), values=list(expenses.values())), use_container_width=True)

elif page == "Investment":
    st.header("Illustrative Investment Projection")
    st.warning("Assumed returns are hypothetical, not guaranteed.")
    monthly = st.number_input("Monthly contribution", min_value=0.0, value=300.0)
    rate = st.number_input("Assumed annual return (%)", min_value=0.0, value=7.0)
    years = st.number_input("Years", min_value=1, value=10)
    if st.button("Calculate"):
        st.metric("Contributed", f"{currency} {monthly * years * 12:,.2f}")
        st.metric("Illustrative value", f"{currency} {fv(monthly, rate, years):,.2f}")
    q = st.text_input("Ask investment education question")
    if st.button("Explain investment") and q:
        ans, refs = rag(q, "investment")
        st.markdown(ans)
        if refs:
            st.dataframe(pd.DataFrame(refs), use_container_width=True)

elif page == "Loans":
    st.header("Loan EMI Calculator")
    p = st.number_input("Principal", min_value=1.0, value=200000.0)
    r = st.number_input("Annual rate (%)", min_value=0.0, value=8.0)
    n = st.number_input("Months", min_value=1, value=60)
    if st.button("Calculate EMI"):
        e, total, interest = emi_calc(p, r, int(n))
        a, b, c = st.columns(3)
        a.metric("Monthly EMI", f"{currency} {e:,.2f}")
        b.metric("Total payment", f"{currency} {total:,.2f}")
        c.metric("Interest", f"{currency} {interest:,.2f}")

elif page == "Tax":
    st.header("Tax Education")
    st.warning("Tax depends on tax residence and tax year; this is not a tax calculator.")
    q = st.text_area("Tax question")
    if country in ("ALL", "GLOBAL"):
        st.info("Select a specific country for tax questions.")
    if st.button("Search tax knowledge") and q and country not in ("ALL", "GLOBAL"):
        ans, refs = rag(q, "taxation")
        st.markdown(ans)
        if refs:
            st.dataframe(pd.DataFrame(refs), use_container_width=True)

elif page == "Insurance":
    st.header("Insurance Coverage Gap")
    existing = st.number_input("Existing cover", min_value=0.0, value=100000.0)
    target = st.number_input("Your selected target", min_value=0.0, value=250000.0)
    st.metric("Difference", f"{currency} {max(0, target - existing):,.2f}")
    st.caption("Not a recommended coverage amount.")
    q = st.text_input("Insurance question")
    if st.button("Explain insurance") and q:
        ans, refs = rag(q, "insurance")
        st.markdown(ans)
        if refs:
            st.dataframe(pd.DataFrame(refs), use_container_width=True)

elif page == "Retirement":
    st.header("Retirement Projection")
    corpus = st.number_input("Current corpus", min_value=0.0, value=10000.0)
    monthly = st.number_input("Monthly contribution", min_value=0.0, value=400.0)
    rate = st.number_input("Assumed annual return (%)", min_value=0.0, value=6.0)
    years = st.number_input("Years", min_value=1, value=20)
    if st.button("Calculate retirement"):
        value = corpus * (1 + rate / 1200) ** (years * 12) + fv(monthly, rate, years)
        st.metric("Illustrative corpus", f"{currency} {value:,.2f}")
        st.caption("Excludes inflation, taxes, fees and changing returns.")

elif page == "Financial Goals":
    st.header("Financial Goal Planner")
    target = st.number_input("Target amount", min_value=0.0, value=50000.0)
    current = st.number_input("Current savings", min_value=0.0, value=10000.0)
    years = st.number_input("Years", min_value=1, value=3)
    rate = st.number_input("Assumed annual return (%)", min_value=0.0, value=0.0)
    if st.button("Calculate monthly target"):
        remaining = max(0, target - current)
        n = years * 12
        r = rate / 1200
        monthly = remaining / n if r == 0 else remaining * r / ((1 + r) ** n - 1)
        st.metric("Illustrative monthly saving", f"{currency} {monthly:,.2f}")

st.divider()
st.caption("Educational prototype. Verify current rules with official sources and qualified professionals.")
