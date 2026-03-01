import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import pickle
import os

# ── Gene families ─────────────────────────────────────────────────────────────
GENE_FAMILIES = {
    0: "G-protein coupled receptors",
    1: "Tyrosine kinase",
    2: "Tyrosine phosphatase",
    3: "Synthetase",
    4: "Synthase",
    5: "Ion channel",
    6: "Transcription factor"
}

# ── Cancer association per gene family ────────────────────────────────────────
CANCER_MAP = {
    0: {
        "cancer_associated": True,
        "cancers": ["Melanoma", "Thyroid Cancer", "Small Cell Lung Cancer"],
        "oncogene": True,
        "risk": "Moderate",
        "detail": "GPCR overexpression linked to melanoma and thyroid malignancies.",
        "drug_targets": ["Vismodegib", "Sonidegib"]
    },
    1: {
        "cancer_associated": True,
        "cancers": ["Chronic Myelogenous Leukemia (CML)", "Breast Cancer", "GIST", "Lung Adenocarcinoma"],
        "oncogene": True,
        "risk": "High",
        "detail": "Tyrosine kinase mutations are among the most well-documented cancer drivers. BCR-ABL fusion in CML, HER2 amplification in breast cancer, KIT mutations in GIST.",
        "drug_targets": ["Imatinib (Gleevec)", "Erlotinib", "Trastuzumab (Herceptin)"]
    },
    2: {
        "cancer_associated": True,
        "cancers": ["T-cell Lymphoma", "Colorectal Cancer"],
        "oncogene": False,
        "risk": "Moderate",
        "detail": "Tyrosine phosphatase acts as a tumor suppressor. Loss of function mutations linked to T-cell lymphoma.",
        "drug_targets": ["Investigational PTP inhibitors"]
    },
    3: {
        "cancer_associated": False,
        "cancers": [],
        "oncogene": False,
        "risk": "Low",
        "detail": "Synthetase gene family primarily associated with metabolic disorders, not directly cancer-linked.",
        "drug_targets": []
    },
    4: {
        "cancer_associated": False,
        "cancers": [],
        "oncogene": False,
        "risk": "Low",
        "detail": "Synthase gene family primarily associated with lipid metabolism. Indirect cancer link via chronic inflammation.",
        "drug_targets": []
    },
    5: {
        "cancer_associated": True,
        "cancers": ["Glioblastoma", "Breast Cancer (ion channel subtype)"],
        "oncogene": True,
        "risk": "Moderate",
        "detail": "Certain ion channel dysregulation drives tumor cell migration and invasion, particularly in glioblastoma.",
        "drug_targets": ["Investigational ion channel blockers"]
    },
    6: {
        "cancer_associated": True,
        "cancers": ["Acute Myeloid Leukemia (AML)", "Breast Cancer", "Prostate Cancer"],
        "oncogene": True,
        "risk": "High",
        "detail": "Transcription factor mutations are key drivers of leukemia. MYC, TP53 mutations are hallmark cancer events.",
        "drug_targets": ["Venetoclax", "Enasidenib", "Ivosidenib"]
    }
}

# ── Carrier risk for couple compatibility ─────────────────────────────────────
CARRIER_MAP = {
    0: "Carrier — GPCR variant detected",
    1: "High-Risk Carrier — Tyrosine kinase mutation marker detected",
    2: "Carrier — Tumor suppressor variant detected",
    3: "Non-Carrier — No cancer-linked variant",
    4: "Non-Carrier — No cancer-linked variant",
    5: "Carrier — Ion channel variant detected",
    6: "High-Risk Carrier — Transcription factor mutation marker detected"
}

def get_kmers(sequence, k=6):
    sequence = sequence.upper().replace(' ', '')
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def kmers_to_string(sequence, k=6):
    return ' '.join(get_kmers(sequence, k))

def load_data():
    dfs = []
    base = os.path.dirname(os.path.abspath(__file__))
    for fname in ['human.txt', 'chimpanzee.txt', 'dog.txt']:
        fpath = os.path.join(base, 'archive', fname)
        if os.path.exists(fpath):
            df = pd.read_table(fpath)
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No dataset files found in archive/ folder.")
    data = pd.concat(dfs, ignore_index=True)
    data.columns = [c.strip().lower() for c in data.columns]
    return data

def train_model():
    from sklearn.feature_extraction.text import CountVectorizer
    print("Loading data...")
    data = load_data()
    data['kmer'] = data['sequence'].apply(kmers_to_string)
    X = data['kmer']
    y = data['class']
    print(f"Dataset size: {len(data)} sequences")
    vectorizer = CountVectorizer(ngram_range=(4,4), analyzer='word')
    X_vec = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    print("Training Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Model Accuracy: {acc*100:.2f}%")
    with open('model.pkl', 'wb') as f:
        pickle.dump({'clf': clf, 'vectorizer': vectorizer, 'accuracy': acc}, f)
    print("Model saved to model.pkl")
    return clf, vectorizer, acc

def load_trained_model():
    if not os.path.exists('model.pkl'):
        return train_model()
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['clf'], data['vectorizer'], data['accuracy']

def detect_cancer_mutations(sequence, gc_content):
    """Detect cancer-specific mutation patterns from sequence."""
    flags = []

    # GC hypermethylation — epigenetic silencing of tumor suppressors
    if gc_content > 65:
        flags.append({
            "type": "GC Hypermethylation",
            "severity": "High",
            "detail": "High GC content suggests promoter hypermethylation — a hallmark of tumor suppressor silencing in cancer."
        })

    # AT instability — chromosomal fragility
    if gc_content < 35:
        flags.append({
            "type": "AT-Rich Instability",
            "severity": "Moderate",
            "detail": "AT-rich regions are prone to chromosomal fragility and deletion — common in early-stage oncogenesis."
        })

    # Microsatellite instability markers
    if 'AAAAA' in sequence:
        flags.append({
            "type": "Microsatellite Instability (MSI)",
            "severity": "High",
            "detail": "Poly-A repeats indicate microsatellite instability — a key marker in colorectal, endometrial, and gastric cancers."
        })

    # Poly-T — splicing disruption
    if 'TTTTT' in sequence:
        flags.append({
            "type": "Splicing Disruption Signal",
            "severity": "Moderate",
            "detail": "Poly-T repeat detected — may disrupt RNA splicing, producing aberrant proteins linked to oncogenesis."
        })

    # G-quadruplex — replication stress
    if 'GGGG' in sequence:
        flags.append({
            "type": "G-Quadruplex Formation",
            "severity": "High",
            "detail": "G-quadruplex motif detected — causes replication fork stalling and DNA double-strand breaks, driving genomic instability in cancer."
        })

    # Tandem repeat — CAG repeat expansion (Huntington-like, also seen in some cancers)
    if 'CAGCAGCAG' in sequence:
        flags.append({
            "type": "CAG Repeat Expansion",
            "severity": "High",
            "detail": "Trinucleotide repeat expansion detected — associated with spinocerebellar ataxias and certain leukemias."
        })

    # TP53 hotspot-like pattern
    if 'ATGATG' in sequence and gc_content > 50:
        flags.append({
            "type": "TP53 Hotspot-like Pattern",
            "severity": "Critical",
            "detail": "Sequence pattern resembles TP53 mutation hotspot — TP53 is mutated in over 50% of all human cancers."
        })

    return flags

def analyze_sequence(sequence):
    """Full cancer-focused analysis of a DNA sequence."""
    clf, vectorizer, acc = load_trained_model()
    sequence = sequence.upper().replace(' ', '').replace('\n', '')
    valid_bases = set('ATGCUN')
    invalid = [c for c in sequence if c not in valid_bases]

    if len(sequence) < 10:
        return {"error": "Sequence too short. Please enter at least 10 bases."}

    # K-mer classification
    kmers = get_kmers(sequence)
    kmer_str = ' '.join(kmers)
    X = vectorizer.transform([kmer_str])
    pred_class = int(clf.predict(X)[0])
    proba = clf.predict_proba(X)[0]
    confidence = float(max(proba) * 100)

    # All confidences
    all_confidences = [
        {"family": GENE_FAMILIES.get(i, f"Family {i}"), "confidence": round(float(proba[i]) * 100, 2)}
        for i in range(len(proba))
    ]

    # Base composition
    base_count = Counter(sequence)
    total = len(sequence)
    gc_content = (base_count.get('G', 0) + base_count.get('C', 0)) / total * 100

    # Cancer mutation detection
    mutation_flags = detect_cancer_mutations(sequence, gc_content)
    if invalid:
        mutation_flags.append({
            "type": "Sequencing Noise",
            "severity": "Low",
            "detail": f"Unknown bases detected: {set(invalid)} — may indicate sequencing error."
        })

    # Cancer info for predicted class
    cancer_info = CANCER_MAP.get(pred_class, CANCER_MAP[3])

    # Overall risk score
    severity_scores = {"Critical": 40, "High": 25, "Moderate": 15, "Low": 5}
    mutation_score = sum(severity_scores.get(f["severity"], 0) for f in mutation_flags)
    base_risk = (1 - max(proba)) * 40
    risk_score = min(100, int(base_risk + mutation_score + (20 if cancer_info["cancer_associated"] else 0)))

    # Status
    if risk_score >= 75:
        status = "Critical Risk"
    elif risk_score >= 50:
        status = "High Risk"
    elif risk_score >= 30:
        status = "Moderate Risk"
    else:
        status = "Low Risk"

    # Codon frequency
    codons = [sequence[i:i+3] for i in range(0, len(sequence)-2, 3) if len(sequence[i:i+3]) == 3]
    codon_counts = Counter(codons)
    top_codons = [
        {"codon": codon, "count": count, "frequency": round(count/len(codons)*100, 2)}
        for codon, count in codon_counts.most_common(10)
    ] if codons else []

    return {
        "sequence_length": len(sequence),
        "gc_content": round(gc_content, 2),
        "predicted_gene_family": GENE_FAMILIES.get(pred_class, "Unknown"),
        "gene_family_id": pred_class,
        "confidence": round(confidence, 2),
        "all_confidences": all_confidences,
        "status": status,
        "risk_score": risk_score,
        "cancer_associated": cancer_info["cancer_associated"],
        "associated_cancers": cancer_info["cancers"],
        "oncogene": cancer_info["oncogene"],
        "cancer_detail": cancer_info["detail"],
        "drug_targets": cancer_info["drug_targets"],
        "carrier_status": CARRIER_MAP.get(pred_class, "Non-Carrier"),
        "mutation_flags": mutation_flags if mutation_flags else [],
        "base_composition": {k: round(v/total*100, 2) for k, v in base_count.items()},
        "kmers_analyzed": len(kmers),
        "top_codons": top_codons
    }

def compare_sequences(seq1, seq2):
    """Cancer carrier compatibility between two sequences."""
    seq1 = seq1.upper().replace(' ', '').replace('\n', '')
    seq2 = seq2.upper().replace(' ', '').replace('\n', '')

    if len(seq1) < 10 or len(seq2) < 10:
        return {"error": "Both sequences must be at least 10 bases long."}

    # K-mer similarity
    kmers1 = set(get_kmers(seq1))
    kmers2 = set(get_kmers(seq2))
    intersection = kmers1 & kmers2
    union = kmers1 | kmers2
    similarity_pct = round(len(intersection) / len(union) * 100, 2) if union else 0

    r1 = analyze_sequence(seq1)
    r2 = analyze_sequence(seq2)

    both_cancer = r1.get('cancer_associated') and r2.get('cancer_associated')
    same_family = r1.get('gene_family_id') == r2.get('gene_family_id')
    combined_risk = (r1.get('risk_score', 0) + r2.get('risk_score', 0)) / 2

    # Offspring cancer risk assessment
    if both_cancer and same_family:
        offspring_risk = "Critical — Both partners carry mutations in the same cancer-linked gene family. Hereditary cancer risk in offspring is significantly elevated."
        compat_score = 15
        recommendation = "Strongly recommend genetic counseling before family planning. Consider preimplantation genetic testing (PGT)."
    elif both_cancer:
        offspring_risk = "High — Both partners carry cancer-associated genetic markers from different pathways. Compound hereditary risk possible."
        compat_score = 35
        recommendation = "Recommend genetic counseling. Carrier screening for specific cancer genes (BRCA1/2, MLH1) advised."
    elif r1.get('cancer_associated') or r2.get('cancer_associated'):
        offspring_risk = "Moderate — One partner carries a cancer-associated genetic marker. 50% carrier probability in offspring."
        compat_score = 60
        recommendation = "Genetic counseling recommended. Regular cancer screening for both partners advised."
    elif similarity_pct > 70:
        offspring_risk = "Moderate — High genomic similarity detected. May indicate shared genetic background."
        compat_score = 55
        recommendation = "Consult a genetic counselor to rule out consanguinity-related risks."
    else:
        offspring_risk = "Low — Neither partner shows significant cancer-associated genetic markers."
        compat_score = 88
        recommendation = "No significant hereditary cancer risk detected. Routine health screening recommended."

    return {
        "partner1_status": r1.get('status'),
        "partner2_status": r2.get('status'),
        "partner1_gene_family": r1.get('predicted_gene_family'),
        "partner2_gene_family": r2.get('predicted_gene_family'),
        "partner1_carrier": r1.get('carrier_status'),
        "partner2_carrier": r2.get('carrier_status'),
        "partner1_cancers": r1.get('associated_cancers', []),
        "partner2_cancers": r2.get('associated_cancers', []),
        "kmer_similarity": similarity_pct,
        "compatibility_score": compat_score,
        "offspring_cancer_risk": offspring_risk,
        "combined_risk_score": round(combined_risk, 1),
        "recommendation": recommendation,
        "shared_kmers": len(intersection)
    }

if __name__ == '__main__':
    train_model()