# 📊 Cloud-Based Document Analytics and Classification

🔗 **Live App**: [Open the Streamlit Web App](https://cloud-document-analytics-dmyi9nuyu3wjx3adw3qpsj.streamlit.app)

This project is a **cloud-ready, Jupyter-based application** for collecting, searching, sorting, and classifying a large collection of academic documents. It uses arXiv's public API to fetch abstracts and PDFs from multiple categories and applies machine learning for classification.

---

## 🔧 Features

- **Document Collection**  
  Automatically collects thousands of academic abstracts and PDF links from arXiv using web scraping (no manual upload required).

- **Sorting by Title**  
  All fetched documents are sorted by their extracted titles (not filenames).

- **Search with Highlighting**  
  You can search the downloaded PDFs for keywords and display highlighted matches directly in the console.

- **Classification**  
  A Random Forest classifier trained on abstract content categorizes documents into:

  - AI/ML
  - Econometrics
  - General Physics
  - Neuroscience
  - Probability

- **Analytics & Statistics**  
  The notebook prints:
  - Number of documents and unique categories
  - Classification training time
  - Weighted F1 Score on test set and via 5-fold cross-validation
  - Size of all referenced PDFs (even if not downloaded)

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/WalaaTayeh/cloud-document-analytics.git
cd cloud-document-analytics
```

### 2. Install dependencies

Make sure Python 3.10 or higher is installed, then run:

```bash
pip install -r requirements.txt
```

### 3. Run the app locally

To launch the Streamlit interface, run:

```bash
streamlit run app.py
```

Then open your browser and go to:

```
http://localhost:8501
```

### 4. Upload or generate dataset

You can upload your `arxiv_dataset.csv` file through the sidebar in the Streamlit interface.  
Alternatively, you can generate this file by running the Jupyter notebook:

```bash
jupyter notebook research_classifier.ipynb
```

The notebook will scrape data from arXiv and save it as `arxiv_dataset.csv`.

### 5. Download PDFs

After the dataset is loaded, specify how many papers you want to download.  
The app will download those PDFs automatically using the arXiv links and save them in the `downloaded_pdfs/` folder.

### 6. Use search, sort, and classify

- **Search**: Enter a keyword in the search field. The app will look inside PDFs and highlight matches with context.
- **Sort**: Click "Sort by Title" to organize documents alphabetically based on actual document titles (not filenames).
- **Classify**: Click "Run Classifier" to classify documents using a Random Forest model. Results will show a confusion matrix, per-class F1 scores, and top TF-IDF features used in the model.

---

## 📁 Project Structure

```
cloud-document-analytics/
│
├── app.py                      # Streamlit app interface
├── research_classifier.ipynb   # Jupyter notebook for data scraping & ML
├── requirements.txt            # Required Python libraries
├── arxiv_dataset.csv           # Metadata file (can be uploaded or generated)
├── downloaded_pdfs/            # Directory for downloaded PDFs
└── README.md                   # Project documentation
```

---
