from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Load trained model + vectorizer
model = joblib.load("dna_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- IMPORTANT: SAME k-mer function as training ---
def get_kmers(sequence, k=3):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def seq_to_kmers(sequence, k=3):
    return " ".join(get_kmers(sequence, k))

@app.route("/")
def home():
    return "API Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    seq = data.get("DNA_sequence")

    if not seq:
        return jsonify({"error": "No sequence provided"}), 400

    # ---- EXACT SAME PREPROCESSING AS TRAINING ----
    kmers = seq_to_kmers(seq, k=3)
    X = vectorizer.transform([kmers])

    pred = int(model.predict(X)[0])

    return jsonify({
        "DNA_sequence": seq,
        "prediction": pred,
        "meaning": "Coding" if pred == 1 else "Non-Coding"
    })

if __name__ == '__main__':
    app.run(debug=True)
