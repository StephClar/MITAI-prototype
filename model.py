import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import pickle
import os

# ── Gene family labels (0–6) ──────────────────────────────────────────────────
GENE_FAMILIES = {
    0: "G-protein coupled receptors",
    1: "Tyrosine kinase",
    2: "Tyrosine phosphatase",
    3: "Synthetase",
    4: "Synthase",
    5: "Ion channel",
    6: "Transcription factor"
}

# Disorders loosely mapped to gene families (for demo purposes)
DISORDER_MAP = {
    0: "Risk of signaling pathway disorder (e.g., retinitis pigmentosa)",
    1: "Risk of cancer-related mutation (e.g., chronic myelogenous leukemia)",
    2: "Risk of autoimmune disorder",
    3: "Risk of metabolic enzyme deficiency",
    4: "Risk of lipid metabolism disorder",
    5: "Risk of channelopathy (e.g., cystic fibrosis)",
    6: "Risk of transcription-related disorder (e.g., Rett syndrome)"
}

def get_kmers(sequence, k=6):
    """Split DNA sequence into overlapping k-mers (like NLP tokens)."""
    sequence = sequence.upper().replace(' ', '')
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def kmers_to_string(sequence, k=6):
    return ' '.join(get_kmers(sequence, k))

def load_data():
    """Load and combine human, chimp, dog datasets."""
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
    """Train the classifier and save it."""
    from sklearn.feature_extraction.text import CountVectorizer
    
    print("Loading data...")
    data = load_data()
    
    # Convert sequences to k-mer strings
    data['kmer'] = data['sequence'].apply(kmers_to_string)
    
    X = data['kmer']
    y = data['class']
    
    print(f"Dataset size: {len(data)} sequences")
    
    # Vectorize (bag of k-mers)
    vectorizer = CountVectorizer(ngram_range=(4,4), analyzer='word')
    X_vec = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Model Accuracy: {acc*100:.2f}%")
    
    # Save model + vectorizer
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

def analyze_sequence(sequence):
    """Full analysis of a single DNA sequence."""
    clf, vectorizer, acc = load_trained_model()
    
    sequence = sequence.upper().replace(' ', '').replace('\n', '')
    
    # Validate
    valid_bases = set('ATGCUN')
    invalid = [c for c in sequence if c not in valid_bases]
    
    if len(sequence) < 10:
        return {"error": "Sequence too short. Please enter at least 10 bases."}
    
    # K-mer features
    kmers = get_kmers(sequence)
    kmer_str = ' '.join(kmers)
    X = vectorizer.transform([kmer_str])
    
    pred_class = int(clf.predict(X)[0])
    proba = clf.predict_proba(X)[0]
    confidence = float(max(proba) * 100)
    
    # Mutation detection: look for unusual base patterns
    base_count = Counter(sequence)
    total = len(sequence)
    gc_content = (base_count.get('G', 0) + base_count.get('C', 0)) / total * 100
    
    mutations = []
    if gc_content > 65:
        mutations.append(f"High GC content ({gc_content:.1f}%) — possible hypermethylation risk")
    if gc_content < 35:
        mutations.append(f"Low GC content ({gc_content:.1f}%) — possible AT-rich instability")
    if 'TTTTT' in sequence:
        mutations.append("Poly-T repeat detected — possible splicing disruption")
    if 'AAAAA' in sequence:
        mutations.append("Poly-A repeat detected — possible transcription termination signal")
    if 'GGGG' in sequence:
        mutations.append("G-quadruplex motif detected — possible replication stalling")
    if invalid:
        mutations.append(f"Unknown bases detected: {set(invalid)} — sequencing noise possible")
    
    # Risk score (0–100)
    risk_score = min(100, int((1 - max(proba)) * 100 + len(mutations) * 10))
    
    # Classification
    is_normal = risk_score < 30 and len(mutations) == 0
    status = "Normal" if is_normal else ("Low Risk" if risk_score < 50 else ("Moderate Risk" if risk_score < 75 else "High Risk"))
    
    return {
        "sequence_length": len(sequence),
        "gc_content": round(gc_content, 2),
        "predicted_gene_family": GENE_FAMILIES.get(pred_class, "Unknown"),
        "gene_family_id": pred_class,
        "confidence": round(confidence, 2),
        "status": status,
        "risk_score": risk_score,
        "disorder_flag": DISORDER_MAP.get(pred_class, "No disorder pattern detected"),
        "mutations_detected": mutations if mutations else ["No significant mutation patterns detected"],
        "base_composition": {k: round(v/total*100, 2) for k, v in base_count.items()},
        "kmers_analyzed": len(kmers)
    }

def compare_sequences(seq1, seq2):
    """Compare two sequences for couple compatibility."""
    seq1 = seq1.upper().replace(' ', '').replace('\n', '')
    seq2 = seq2.upper().replace(' ', '').replace('\n', '')
    
    if len(seq1) < 10 or len(seq2) < 10:
        return {"error": "Both sequences must be at least 10 bases long."}
    
    # K-mer similarity (Jaccard)
    k = 6
    kmers1 = set(get_kmers(seq1, k))
    kmers2 = set(get_kmers(seq2, k))
    
    intersection = kmers1 & kmers2
    union = kmers1 | kmers2
    jaccard = len(intersection) / len(union) if union else 0
    similarity_pct = round(jaccard * 100, 2)
    
    # Analyze both individually
    r1 = analyze_sequence(seq1)
    r2 = analyze_sequence(seq2)
    
    # Compatibility logic
    same_family = r1.get('gene_family_id') == r2.get('gene_family_id')
    combined_risk = (r1.get('risk_score', 0) + r2.get('risk_score', 0)) / 2
    
    if similarity_pct > 70 and same_family:
        compatibility = "High Risk — Very high genetic similarity may indicate consanguinity or shared disorder risk"
        compat_score = 20
    elif combined_risk > 60:
        compatibility = "Moderate Risk — Both sequences carry disorder patterns; offspring risk elevated"
        compat_score = 45
    elif similarity_pct > 40:
        compatibility = "Low-Moderate Risk — Moderate similarity; recommend genetic counseling"
        compat_score = 65
    else:
        compatibility = "Compatible — Genetic profiles show healthy diversity"
        compat_score = 90
    
    return {
        "sequence_1_status": r1.get('status'),
        "sequence_2_status": r2.get('status'),
        "sequence_1_gene_family": r1.get('predicted_gene_family'),
        "sequence_2_gene_family": r2.get('predicted_gene_family'),
        "kmer_similarity": similarity_pct,
        "compatibility_score": compat_score,
        "compatibility_result": compatibility,
        "shared_kmers": len(intersection),
        "combined_risk_score": round(combined_risk, 1),
        "recommendation": "Consult a genetic counselor for clinical interpretation." 
    }

if __name__ == '__main__':
    train_model()