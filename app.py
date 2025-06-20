import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from pathlib import Path
import requests
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =================== Helper Functions ===================

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text")
        doc.close()
    except Exception as e:
        st.error(f"Error reading {pdf_path.name}: {e}")
    return text

def download_pdfs(df, limit):
    pdf_dir = Path("downloaded_pdfs")
    pdf_dir.mkdir(exist_ok=True)
    saved_files = []
    sample_df = df.sample(n=min(limit, len(df)), random_state=42)
    for idx, row in sample_df.iterrows():
        if row['pdf_url']:
            pdf_path = pdf_dir / f"doc_{idx}.pdf"
            try:
                response = requests.get(row['pdf_url'], timeout=10)
                if response.status_code == 200:
                    with open(pdf_path, 'wb') as f:
                        f.write(response.content)
                    saved_files.append((pdf_path, row['title']))
                    st.success(f"Downloaded: {pdf_path.name}")
                else:
                    st.warning(f"Failed to download: {row['pdf_url']}")
            except Exception as e:
                st.warning(f"Error: {e}")
    return saved_files

def search_keyword_in_pdfs(pdf_files, keyword):
    results = []
    keyword = keyword.lower()
    for pdf_path, title in pdf_files:
        text = extract_text_from_pdf(pdf_path).lower()
        if keyword in text:
            snippet = text[text.find(keyword)-50:text.find(keyword)+50]
            results.append((title, snippet))
    return results

def classify_documents(df):
    vectorizer = TfidfVectorizer(max_features=10000)
    clf = RandomForestClassifier(n_estimators=400)
    pipe = Pipeline([
        ("vec", vectorizer),
        ("clf", clf)
    ])
    df = df.dropna(subset=["abstract", "category"])
    X_train, X_test, y_train, y_test = train_test_split(df['abstract'], df['category'], test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report

# =================== UI ===================

st.set_page_config(page_title="Cloud Document Analyzer", layout="wide")
st.title("📄 Cloud Document Analyzer")

# Upload CSV
st.sidebar.header("📥 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload arxiv_dataset.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.dropna(subset=["pdf_url", "title"], inplace=True)

    st.sidebar.header("🔎 Search Settings")
    keyword = st.sidebar.text_input("Enter a keyword to search in PDFs", value="quantum")
    limit = st.sidebar.slider("Number of PDFs to download", 1, 20, 5)

    if st.sidebar.button("Start Search"):
        with st.spinner("🔽 Downloading and processing PDFs..."):
            pdf_files = download_pdfs(df, limit)
            results = search_keyword_in_pdfs(pdf_files, keyword)

        st.subheader("🔍 Search Results")
        if results:
            for title, snippet in results:
                st.markdown(f"**✅ {title}**\n\n`...{snippet}...`")
        else:
            st.warning("No matching documents found.")

    st.sidebar.header("📊 Dataset Tools")
    if st.sidebar.button("Sort by Title"):
        st.subheader("📄 Sorted Documents")
        sorted_df = df.sort_values(by='title')
        st.dataframe(sorted_df[['title', 'category']].head(10))

    if st.sidebar.button("Run Classifier"):
        with st.spinner("⚙️ Training classifier..."):
            report = classify_documents(df)
        st.subheader("🤖 Classification Report")
        st.json(report)

    st.sidebar.header("📈 Dataset Info")
    st.sidebar.markdown(f"**Total Documents**: {len(df)}")
    st.sidebar.markdown(f"**Unique Categories**: {df['category'].nunique()}")
    st.sidebar.markdown(f"**Unique Titles**: {df['title'].nunique()}")

else:
    st.warning("Please upload your arxiv_dataset.csv file to begin.")
