import joblib

model = joblib.load("dna_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def get_kmers(sequence, k=3):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def seq_to_kmers(sequence, k=3):
    return " ".join(get_kmers(sequence, k))

seq = "GGCTACGACGTGACCGCGGGGCAGGTGCTCGTGACCAACGGCGGCAAGCAGGCGGTGGAGGAGACCTGCGCAA"


kmers = seq_to_kmers(seq)
X = vectorizer.transform([kmers])

print("Prediction:", model.predict(X)[0])
