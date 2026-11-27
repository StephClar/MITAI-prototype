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

1. Clone the repo

git clone https://github.com/StephClar/MITAI-prototype.git
cd MITAI-prototype

2. Create a virtual environment

python3 -m venv venv
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Train the model

python model.py

5. Start the server

python app.py

6. Open the app

Open http://127.0.0.1:5000 in your browser.

7. (Optional) Re-run the model evaluation

python evaluate.py

This rewrites `metrics.json`, which the **Model Evaluation** tab displays.

---

## 🧠 About the AI (shown in the app)

The **About the AI** tab explains which model is used and why. Every analysis also shows a step-by-step **"How the AI analysed this sequence"** panel with that sequence's real numbers: k-mers created, phrases recognised, and how each of the 100 trees voted.

- **Model:** Random Forest classifier (scikit-learn, 100 trees) on k-mer bag-of-words features
- **Predicts:** one of 7 gene families; cancer links, flags and the risk score are rule-based on top
- **Why:** suits a small dataset (~7k sequences) and sparse features, robust, gives class probabilities, trains in seconds without a GPU
- **Next stage:** DNA/protein language models (DNABERT, ESM-2)

## 📊 Model Evaluation

The original random 80/20 split reported ~94% accuracy. Investigating further, we found **data leakage**: the dataset contains 821 exact duplicate sequences, and 61% of test sequences had a ≥90% similar "twin" in the training set (human and chimpanzee genes are near-identical). A random split therefore rewards memorisation.

`evaluate.py` tests the same Random Forest three ways:

| Test method | Accuracy | What it measures |
|---|---|---|
| Random 80/20 split | 93.6% | Original method — optimistic |
| **Similarity-grouped split** (≥80% similar sequences kept together) | **86.7%** | Honest estimate on genuinely new genes |
| Cross-species (train human + chimp, test dog) | 88.3% | Generalisation to an unseen species |
| Always guess the most common family | 30.3% | Baseline |

Weakest family: **Ion channel** (57% recall), most often confused with GPCRs. Tyrosine kinase reaches 100% precision but 79% recall.

All figures are gene-family classification accuracy, not cancer-detection accuracy.

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
├── evaluate.py            # Honest model evaluation (leakage-aware splits)
├── metrics.json           # Evaluation results shown in the app
├── app.py                 # Flask API server
├── index.html             # Frontend UI
├── requirements.txt
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
Built by the **Entwined Edit** team. MITAI is the prototype behind our Entwined Edit startup pitch.

---

## 🥇 Recognition

| Event | Result |
|-------|--------|
| Anna University Startup Pitch | 🥇 1st place |
| LICET eCirkle | 🥉 3rd place |
| Anna University Incubation Cell | Selected for incubation |


---

## 📌 Dataset

The model is trained on the public **DNA Sequence Dataset** on Kaggle, published by Nagesh Singh Chauhan (originally from his GitHub repository [nageshsinghc4/DNA-Sequence-Machine-learning](https://github.com/nageshsinghc4/DNA-Sequence-Machine-learning)). It is widely used in DNA-classification tutorials and research papers.

**What the DNA is:** each row is the **coding sequence** of a real gene — the part of the gene that is translated into a protein — written as A/T/G/C letters, with a label for its gene family (0–6).

**Where it comes from:** no DNA was collected by our team, and there is no patient data. These are reference gene sequences from three species' published genomes, made publicly available for research. The dataset author does not document the exact upstream database; public gene sequences like these are normally drawn from genome databases such as NCBI or Ensembl.

| File | Species | Sequences |
|------|---------|-----------|
| `archive/human.txt` | Human (*Homo sapiens*) | 4,380 |
| `archive/chimpanzee.txt` | Chimpanzee (*Pan troglodytes*) | 1,682 |
| `archive/dog.txt` | Dog (*Canis familiaris*) | 820 |
| **Total** | | **6,882** |

**Gene family labels:** 0 G-protein coupled receptors · 1 Tyrosine kinase · 2 Tyrosine phosphatase · 3 Synthetase · 4 Synthase · 5 Ion channel · 6 Transcription factor

**Why three species?** Human and chimpanzee genes are about 98–99% identical, while dog is more distant. This lets us test whether the model generalises to a species it has never seen (see Model Evaluation).

In the full Entwined Edit pipeline, this dataset stands in for the sequences an Oxford Nanopore device would produce from a real sample.
