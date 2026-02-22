# 🧬 MITAI — Genomic Intelligence Platform

> AI-powered DNA sequence analysis for faster, accessible genetic diagnostics.

**Disclaimer:** This is a computational proof-of-concept prototype built for a hackathon. It is not intended for clinical use. All outputs are for demonstration purposes only.

---

## The Problem

Traditional gene anomaly detection takes **4–12 weeks** and requires expensive equipment and deep technical expertise — making it inaccessible to small clinics, labs, and researchers.

## Our Solution

MITAI reduces this to **~2 days** using an AI pipeline that:
- Processes DNA/RNA sequences using NLP-style techniques
- Classifies gene families and flags disorder patterns
- Detects mutation-like variations automatically
- Generates structured diagnostic-style reports
- Compares two genetic profiles for couple compatibility

---

## Real-World Pipeline

```
DNA Sample → Oxford Nanopore Sequencer → fast5 file
→ Convert to FastQ → pLM (Protein Language Model)
→ Anomaly Detection → Diagnostic Report
```

In this prototype, the Nanopore step is simulated using a pre-existing labeled DNA dataset.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Model | Random Forest Classifier |
| Sequence Processing | K-mer tokenization (NLP for DNA) |
| Backend | Python, Flask |
| Frontend | HTML, CSS, JavaScript |
| Dataset | Human, Chimpanzee & Dog DNA sequences (gene family classification) |

---

## Features

- **Sequence Analysis** — paste any DNA/RNA sequence and get:
  - Gene family classification (7 families)
  - Risk score (0–100)
  - GC content analysis
  - Mutation pattern detection
  - Genetic disorder flag

- **Couple Compatibility** — compare two sequences and get:
  - K-mer similarity score
  - Compatibility assessment
  - Combined risk score

---

## Project Structure


---


## 📌 Note on Dataset

This prototype uses the [DNA Sequence Dataset](https://www.kaggle.com/datasets/nageshsingh/dna-sequence-dataset) from Kaggle, containing labeled DNA sequences from human, chimpanzee, and dog genomes across 7 gene families.
