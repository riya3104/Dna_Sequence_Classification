import React, { useState } from "react";

function DNAForm() {
  const [sequence, setSequence] = useState("");
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  // ----------- CLEAN + SANITIZE INPUT -----------------
  const handleChange = (e) => {
    let value = e.target.value;

    value = value.toUpperCase();        // convert lowercase → uppercase
    value = value.replace(/[^ATCG]/g, ""); // remove all characters except A,T,C,G

    setSequence(value);
    setError("");
    setResult(null);
  };

  // -------------- VALIDATE DNA -------------------------
  const validateDNA = (str) => {
    if (str.length < 60) {
      setError("Please enter at least 60 nucleotides for accurate prediction.");
      return false;
    }
    return true;
  };

  // -------------- SEND REQUEST TO BACKEND --------------
  const handlePredict = async () => {
    setError("");

    if (!validateDNA(sequence)) return;

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ DNA_sequence: sequence }),
      });

      const data = await response.json();

      if (data.error) {
        setError(data.error);
        return;
      }

      setResult(data);

    } catch (err) {
      console.error(err);
      setError("API request failed. Make sure Flask backend is running.");
    }
  };

  return (
    <div style={{ width: "60%", margin: "auto", marginTop: "50px", textAlign: "center" }}>
      <h1>DNA Coding / Non-Coding Predictor</h1>

      <textarea
        value={sequence}
        onChange={handleChange}
        rows="5"
        style={{ width: "100%", padding: "10px", marginTop: "20px" }}
        placeholder="Paste DNA sequence (A,T,C,G only)..."
      ></textarea>

      {error && (
        <p style={{ color: "red", fontWeight: "bold" }}>{error}</p>
      )}

      <button
        onClick={handlePredict}
        style={{
          marginTop: "20px",
          padding: "10px 25px",
          fontSize: "16px",
          cursor: "pointer",
        }}
      >
        Predict
      </button>

      {result && (
        <div style={{ marginTop: "30px" }}>
          <h2>Prediction Result</h2>
          <p><strong>Input DNA (cleaned):</strong> {result.DNA_sequence}</p>
          <p>
            <strong>Prediction:</strong>{" "}
            {result.prediction === 1 ? "Coding" : "Non-Coding"}
          </p>
        </div>
      )}
    </div>
  );
}

export default DNAForm;
