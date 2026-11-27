"""
Honest evaluation of the MITAI gene-family classifier.

Why this exists:
The original 80/20 random split reported ~94% accuracy. But many sequences in
the dataset are exact copies or near-copies of each other (human and chimp genes
are very similar, and the human file alone has 751 exact duplicates). With a
random split, a test sequence often has a near-twin in the training data, so the
model is partly "remembering" instead of generalising. That inflates accuracy.

This script compares three ways of testing:
  1. Random split          - the original method (optimistic)
  2. Similarity-grouped    - sequences that are >= 80% similar are kept on the
                             SAME side of the split, so no near-twins leak
  3. Cross-species         - train on human + chimp, test on dog only

Run:   python evaluate.py
Output: metrics.json (shown in the app's "Model Evaluation" tab)
"""
import json
import os
from datetime import date

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from model import GENE_FAMILIES, get_kmers, kmers_to_string

BASE = os.path.dirname(os.path.abspath(__file__))
SIMILARITY_THRESHOLD = 0.8   # k-mer Jaccard similarity that counts as a "near-twin"
SEED = 42
CLASSES = list(range(7))


def load_with_species():
    dfs = []
    for species in ['human', 'chimpanzee', 'dog']:
        df = pd.read_table(os.path.join(BASE, 'archive', f'{species}.txt'))
        df.columns = [c.strip().lower() for c in df.columns]
        df['species'] = species
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data['kmer'] = data['sequence'].apply(kmers_to_string)
    return data


def train_and_score(train, test):
    """Same model settings as model.py, trained and tested on the given split."""
    vec = CountVectorizer(ngram_range=(4, 4), analyzer='word')
    clf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    clf.fit(vec.fit_transform(train['kmer']), train['class'])
    pred = clf.predict(vec.transform(test['kmer']))
    return pred


def summary(test, pred):
    return {
        "test_size": int(len(test)),
        "accuracy": round(float(accuracy_score(test['class'], pred)) * 100, 1),
        "macro_f1": round(float(f1_score(test['class'], pred, average='macro')) * 100, 1),
    }


def jaccard_matrix(sequences):
    """Pairwise similarity: shared 6-mers / all distinct 6-mers, for every pair."""
    vec = CountVectorizer(analyzer=lambda s: set(get_kmers(s)), binary=True)
    m = vec.fit_transform(sequences).astype(np.float32)
    inter = (m @ m.T).toarray()
    n = np.asarray(m.sum(axis=1)).ravel()
    return inter / (n[:, None] + n[None, :] - inter)


def main():
    print("Loading data...")
    data = load_with_species()
    print(f"{len(data)} sequences")

    print("Computing sequence similarity (takes ~30 s)...")
    sim = jaccard_matrix(data['sequence'])

    # ── Leakage check on the original random split ────────────────────────────
    train, test = train_test_split(data, test_size=0.2, random_state=SEED)
    nearest = sim[np.ix_(test.index, train.index)].max(axis=1)
    leakage = {
        "exact_duplicates_in_dataset": int(data.duplicated('sequence').sum()),
        "test_with_identical_train_seq_pct": round(float((nearest >= 0.999).mean()) * 100, 1),
        "test_with_90pct_similar_train_seq_pct": round(float((nearest >= 0.9).mean()) * 100, 1),
    }

    print("1/3 Random split...")
    random_split = summary(test, train_and_score(train, test))

    print("2/3 Similarity-grouped split...")
    n_groups, groups = connected_components(csr_matrix(sim >= SIMILARITY_THRESHOLD), directed=False)
    tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
                          .split(data, groups=groups))
    g_train, g_test = data.iloc[tr_idx], data.iloc[te_idx]
    g_pred = train_and_score(g_train, g_test)
    grouped = summary(g_test, g_pred)
    grouped["groups"] = int(n_groups)
    majority = g_train['class'].mode()[0]
    grouped["majority_baseline_accuracy"] = round(float((g_test['class'] == majority).mean()) * 100, 1)

    p, r, f, s = precision_recall_fscore_support(g_test['class'], g_pred, labels=CLASSES, zero_division=0)
    per_class = [
        {"class": c, "family": GENE_FAMILIES[c], "precision": round(float(p[i]) * 100, 1),
         "recall": round(float(r[i]) * 100, 1), "f1": round(float(f[i]) * 100, 1), "support": int(s[i])}
        for i, c in enumerate(CLASSES)
    ]
    cm = confusion_matrix(g_test['class'], g_pred, labels=CLASSES).tolist()

    print("3/3 Cross-species (train human+chimp, test dog)...")
    cs_train, cs_test = data[data.species != 'dog'], data[data.species == 'dog']
    cross_species = summary(cs_test, train_and_score(cs_train, cs_test))

    metrics = {
        "generated": date.today().isoformat(),
        "dataset_size": int(len(data)),
        "similarity_threshold_pct": int(SIMILARITY_THRESHOLD * 100),
        "leakage": leakage,
        "random_split": random_split,
        "grouped_split": grouped,
        "cross_species": cross_species,
        "per_class": per_class,
        "confusion_matrix": cm,
        "class_names": [GENE_FAMILIES[c] for c in CLASSES],
    }
    with open(os.path.join(BASE, 'metrics.json'), 'w') as fh:
        json.dump(metrics, fh, indent=2)

    print("\n── Results ──")
    print(f"Random split:            {random_split['accuracy']}% accuracy (optimistic)")
    print(f"Similarity-grouped:      {grouped['accuracy']}% accuracy (honest)")
    print(f"Cross-species (dog):     {cross_species['accuracy']}% accuracy")
    print(f"Always-guess-majority:   {grouped['majority_baseline_accuracy']}%")
    print("Saved metrics.json")


if __name__ == '__main__':
    main()
