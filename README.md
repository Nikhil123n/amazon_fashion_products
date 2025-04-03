# Amazon Fashion Product Similarity Search

This project is a product recommendation system designed to find visually and semantically similar fashion products from Amazon's 30K dataset using attributes like **brand, price, weight, color, and textual product names**.

---

## Objective

Create a microservice that can identify top-N most similar fashion products from a dataset of 30K Amazon fashion items using:
- Attributes like brand, product_name, price, rating, weight, and color
- Cosine similarity on a hybrid feature matrix
- Random Forest for imputing missing `weight` values (critical but noisy feature)
- A FastAPI server exposed via `/find_similar_products` endpoint
- Dockerized container ready for deployment on Kubernetes

---

## Modeling Journey & Backstory

### Why weight imputation was critical?
The `weight` field was severely inconsistent — many entries were in `kg`, `g`, or even strings like `2-pack`, with ~48% missing values and a placeholder value of `999999999`. Rather than dropping such a large portion, we:
- Preprocessed units (e.g., `2.5kg → 2500g`)
- Removed placeholder outliers
- Imputed missing values using **Random Forest Regressor** (because:
  - It can model nonlinear relations (unlike linear models)
  - It is more robust to outliers than Decision Trees
  - It performs better than Decision Tree in R² and RMSE on this imputation task)

### Why not use Decision Tree?
Tried Decision Tree:
- Without filtering outliers, R² was ~0.36, RMSE ~5000
- With filtering: R² dropped to ~0.21, indicating poor generalization
- Random Forest had stable R² ~0.38+ and better RMSE, so got stuck with it

---

### Text + Feature Engineering Decisions

- `brand`: Cleaned for punctuation and missing values. One-hot encoded.
- `product_name`: Preprocessed with:
  - `contractions` to expand phrases like "can't" to "cannot"
  - Regular expressions to clean digits, punctuations
  - `nltk.stopwords` to remove noise
  - `SnowballStemmer` to stem similar words (helping TF-IDF)
- `colour`: Multi-hot encoded using `MultiLabelBinarizer` after handling messy strings like `["black|red"]`

### Final Feature Matrix

```python
Final Matrix = [
  OneHotEncoded(brand),
  MinMaxScaled([price, rating, imputed_weight]),
  TFIDF(product_name),
  MultiLabelEncoded(colour)
]
```

Then used **Cosine Similarity** on this matrix.

To handle ties in similarity score, implemented a **secondary sorting based on price**.

---

## How to Run

### Clone & Setup

```bash
git clone https://github.com/yourusername/product-similarity-app.git
cd product-similarity-app
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Run FastAPI Server

```bash
uvicorn main:app --reload
```

Access it at [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Docker Support

```bash
# Build
docker build -t product-similarity-app .

# Run
docker run -p 8000:8000 product-similarity-app
```

See [README_DEPLOYMENT.md](README_DEPLOYMENT.md) for Kubernetes YAML setup.

---

## Example API Call

```
GET /find_similar_products?product_id=252705c23c50c7988e9713ab92f7d0d3&num_similar=5
```

Returns a JSON list of top 5 similar product IDs based on hybrid feature matrix.

---

## Project Structure

```
.
├── data/                    # Contains LDJSON Amazon fashion dataset
├── main.py                  # FastAPI microservice logic
├── model_utils.py           # Weight imputation, feature matrix creation, cosine search
├── preprocessing.py         # Feature cleaning helpers (product_name, brand, etc.)
├── requirements.txt
├── Dockerfile
├── deployment.yaml          # Kubernetes deployment manifest
├── service.yaml             # Kubernetes service manifest
├── amazon_similarity.ipynb  # Jupyter Notebook implementation of similarity
├── README.md
└── README_DEPLOYMENT.md     # Deployment documentation (Docker + K8s)
```

---

## Authors

Built with by a developer obsessed with solving real-world data messiness.

---