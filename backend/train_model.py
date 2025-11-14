import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.svm import LinearSVC
import joblib
import os
import numpy as np

RANDOM_STATE = 42

# ---------------------------
# 1) Load dataset
# ---------------------------
csv_path = r"C:\Users\RIYA\OneDrive\Documents\DNA sequence analysis\Coding_NonCoding_DNA_Sequences.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found at: {csv_path}")

df = pd.read_csv(csv_path)

# Keep only columns we need (rename if necessary)
expected_cols = ['DNA_sequence', 'Target']
if not set(expected_cols).issubset(df.columns):
    print("Found columns:", df.columns.tolist())
    raise ValueError("Dataset does not contain required columns: 'DNA_sequence' and 'Target'")

df = df[['DNA_sequence', 'Target']].copy()
df.dropna(subset=['DNA_sequence', 'Target'], inplace=True)

# Make sure sequences are uppercase and contain only ATCG
df['DNA_sequence'] = df['DNA_sequence'].astype(str).str.upper().str.replace(r'[^ATCG]', '', regex=True)

print("Original label counts:")
print(df['Target'].value_counts())

# ---------------------------
# 2) Balance dataset (downsample majority class)
# ---------------------------
count0 = (df['Target'] == 0).sum()
count1 = (df['Target'] == 1).sum()
print(f"Counts before balancing -> 0: {count0}, 1: {count1}")

# If one class is larger, downsample it to minority size
if count0 > count1:
    df_major = df[df['Target'] == 0]
    df_minor = df[df['Target'] == 1]
    df_major_down = df_major.sample(n=len(df_minor), random_state=RANDOM_STATE)
    df_balanced = pd.concat([df_major_down, df_minor]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
elif count1 > count0:
    df_major = df[df['Target'] == 1]
    df_minor = df[df['Target'] == 0]
    df_major_down = df_major.sample(n=len(df_minor), random_state=RANDOM_STATE)
    df_balanced = pd.concat([df_major_down, df_minor]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
else:
    df_balanced = df.copy()

print("Label counts after balancing:")
print(df_balanced['Target'].value_counts())

# ---------------------------
# 3) Create k-mers (k = 6)
# ---------------------------
K = 6

def get_kmers(sequence: str, k: int = K):
    if not isinstance(sequence, str) or len(sequence) < k:
        return []
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# create space-joined kmers string for vectorizer
df_balanced['kmers'] = df_balanced['DNA_sequence'].apply(lambda s: " ".join(get_kmers(s, K)))

# drop rows where kmers empty (too short sequences)
df_balanced = df_balanced[df_balanced['kmers'].str.len() > 0].reset_index(drop=True)
print("After k-mer creation, sample count:", len(df_balanced))

# ---------------------------
# 4) Vectorize (CountVectorizer)
# ---------------------------
cv = CountVectorizer(analyzer='word')  # each "word" is a k-mer because we joined them with spaces
X = cv.fit_transform(df_balanced['kmers'])
y = df_balanced['Target'].values

print("Feature matrix shape:", X.shape)

# ---------------------------
# 5) Train/test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

# ---------------------------
# 6) Train classifier (LinearSVC)
# ---------------------------
clf = LinearSVC(max_iter=20000, random_state=RANDOM_STATE)
print("Training LinearSVC...")
clf.fit(X_train, y_train)

# ---------------------------
# 7) Evaluate
# ---------------------------
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}\n")
print("Classification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------
# 8) Save model + vectorizer
# ---------------------------
joblib.dump(clf, "dna_model.pkl")
joblib.dump(cv, "vectorizer.pkl")
print("\nSaved dna_model.pkl and vectorizer.pkl to current folder.")

# Small sanity check — quick prediction on a couple of real examples
def quick_test(sequence):
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    km = " ".join(get_kmers(seq, K))
    if not km:
        return None
    return int(clf.predict(cv.transform([km]))[0])

examples = [
    # coding-like (from dataset)
    "GGCTACGACGTGACCGCGGGGCAGGTGCTCGTGACCAACGGCGGCAAGCAGGCGGTGGAGGAGACCTGCGCGACCATCCTCGATCCCGGCGACGAGGTGCTGCTGCCGGCCCC",
    # non-coding-like (sample)
    "TTGTACTTTTATGATAATTTGGTATATGCGTATACATTATTTGTGACATACATTTGTGAATAATGATATTAG"
]

print("\nQuick sanity predictions (1=coding,0=non-coding):")
for ex in examples:
    print(ex[:60] + "... ->", quick_test(ex))
