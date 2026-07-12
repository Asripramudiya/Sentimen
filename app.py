from pathlib import Path
import joblib
import numpy as np
from flask import Flask, render_template, request
from preprocessing import preprocess_text

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

model = joblib.load(BASE_DIR / "model_sentimen.joblib")
tfidf = joblib.load(BASE_DIR / "tfidf.joblib")

def get_confidence(model, vector):
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(vector).max())
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(vector))
        scores = scores[0] if scores.ndim > 1 else scores
        if np.ndim(scores) == 0:
            scores = np.array([-scores, scores])
        scores = scores - np.max(scores)
        probs = np.exp(scores) / np.exp(scores).sum()
        return float(np.max(probs))
    return None

@app.route("/", methods=["GET", "POST"])
def index():
    review = ""
    prediction = None
    confidence = None
    processed = ""
    error = None

    if request.method == "POST":
        review = request.form.get("review", "").strip()
        if not review:
            error = "Masukkan review terlebih dahulu."
        else:
            processed = preprocess_text(review)

            if not processed:
                error = "Tidak dapat diprediksi"
                confidence = 0
            else:
                vector = tfidf.transform([processed])
                # Jika tidak ada kata yang dikenal TF-IDF
                if vector.nnz == 0:
                    prediction = "Tidak dapat diprediksi"
                    confidence = 0
                else:
                    prediction = str(model.predict(vector)[0])
                
                    score = get_confidence(model, vector)
                    confidence = round(score * 100, 2) if score is not None else None

    return render_template(
        "index.html",
        review=review,
        prediction=prediction,
        confidence=confidence,
        processed=processed,
        error=error,
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
