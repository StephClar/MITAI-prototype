# 🧬 MITAI — Cancer Mutation Detection Platform

> AI-powered DNA sequence analysis for early detection of cancer-associated genetic mutations.

**⚠️ Disclaimer:** This is a computational proof-of-concept prototype built for a hackathon. It is not intended for clinical use. All outputs are for demonstration purposes only.

---

## The Problem

Cancer is fundamentally a genetic disease — driven by mutations in specific genes. Detecting these mutations currently takes **4–12 weeks** using conventional methods, requires expensive proprietary equipment, and is accessible only to large hospitals and research institutions.

Small clinics, rural healthcare centers, and labs in developing regions have no access to these tools.

##  Our Solution

MITAI is an AI-powered genomic platform that analyzes DNA sequences to detect cancer-associated mutation patterns. It focuses on **tyrosine kinase pathway mutations** — one of the most well-documented cancer genetic markers, linked to CML, Breast Cancer, GIST, and Lung Adenocarcinoma.

MITAI reduces detection time to **~2 days** and is built entirely on open-source tools — making it affordable and accessible.

---

## 🔬 Real-World Pipeline

```
DNA Sample → Oxford Nanopore Sequencer → fast5 file
→ Convert to FastQ → pLM (Protein Language Model)
→ Cancer Mutation Detection → Oncology Report
```

In this prototype, the Nanopore sequencing step is simulated using a pre-existing labeled DNA dataset. The AI analysis step — the core of our pipeline — is fully implemented.

---

## 🎯 Focus Area

**Primary:** Cancer mutation detection — specifically tyrosine kinase pathway abnormalities linked to:
- Chronic Myelogenous Leukemia (CML)
- Breast Cancer (HER2)
- Gastrointestinal Stromal Tumor (GIST)
- Lung Adenocarcinoma

**Secondary:** Hereditary cancer risk screening — couple compatibility analysis to assess offspring cancer risk based on shared genetic markers.

**Roadmap:** Neurological disorders (Transcription factor mutations — Rett syndrome, AML)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Model | Random Forest Classifier |
| Sequence Processing | K-mer tokenization (NLP applied to DNA) |
| Backend | Python, Flask |
| Frontend | HTML, CSS, JavaScript |
| PDF Reports | jsPDF |
| Dataset | Real human, chimpanzee & dog genomic sequences (gene family classification) |

---

##  What the Prototype Does

### Cancer Mutation Analysis
- Takes raw DNA sequence as input
- Classifies into gene family (7 families) using k-mer tokenization
- Detects cancer-associated mutation patterns:
  - GC Hypermethylation — tumor suppressor silencing
  - Microsatellite Instability (MSI) — colorectal/gastric cancer marker
  - G-Quadruplex Formation — replication stress, genomic instability
  - TP53 Hotspot-like patterns — mutated in 50%+ of all cancers
  - Splicing Disruption Signals
- Generates oncogenic risk score (0–100)
- Flags associated cancer types and known drug targets (Imatinib, Trastuzumab etc.)
- Downloads a full Oncology Report as PDF

### Couple Hereditary Cancer Screening
- Compares two DNA sequences
- Assesses carrier status for cancer-linked gene families
- Estimates offspring hereditary cancer risk
- Provides clinical recommendation

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/StephClar/MITAI-prototype.git
cd MITAI-prototype
```

**2. Install dependencies**
```bash
pip install flask flask-cors scikit-learn pandas numpy
```

**3. Train the model** (one time only, ~2 minutes)
```bash
python model.py
```

**4. Start the server**
```bash
python app.py
```

**5. Open the app**

Double-click `index.html` or open it in your browser.

---

## 📁 Project Structure

```
MITAI-prototype/
├── archive/
│   ├── human.txt          # Real human genomic sequences
│   ├── chimpanzee.txt     # Chimpanzee genomic sequences
│   ├── dog.txt            # Dog genomic sequences
│   └── example_dna.fa     # FASTA example
├── model.py               # ML pipeline, cancer mutation detection logic
├── app.py                 # Flask API server
├── index.html             # Frontend UI
└── README.md
```

---

## 🏥 Why Not a Clinical Dataset?

Labeled sequence-to-disease datasets are not publicly available due to patient privacy regulations (HIPAA, GDPR). They exist only in clinical databases like **NCBI ClinVar** and **COSMIC** which require institutional access.

We use real genomic sequences with gene family classification, and map cancer associations based on published biological literature. This is the standard approach in early-stage academic genomics research.

In production, the disorder mapping would be replaced by a model trained on ClinVar or COSMIC patient data.

---

## 🔑 Why Tyrosine Kinase?

Tyrosine kinase mutations are among the **most well-documented cancer drivers** in existence. The drug Imatinib (Gleevec) — one of the most successful cancer treatments ever developed — was designed specifically to target BCR-ABL tyrosine kinase mutations in CML patients. This is real, peer-reviewed, clinically validated biology.

---

## 🏆 Competitors

| Company | Approach | Limitation |
|---------|----------|------------|
| Tempus | Genomic AI for cancer treatment | Expensive, large hospitals only |
| Freenome | Blood-based cancer detection | Proprietary, high cost |
| DELFI Diagnostics | DNA fragment analysis | Complex setup |
| Guardant Health | Liquid biopsy | FDA process, costly |
| **MITAI** |**Nanopore + open-source AI** | **Affordable, accessible, portable** |

---

## 👥 Team
Made for TNWISE Hackathon 


---

## 📌 Dataset

This prototype uses real genomic sequence data from human, chimpanzee, and dog genomes, labeled by gene family (0–6), sourced from the UCI ML genomics dataset.
