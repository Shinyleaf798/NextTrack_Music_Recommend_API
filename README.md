# NextTrack: Music Recommendation API 🎵

> A Stateless Content-Based Filtering Prototype for Music Discovery

NextTrack is a highly optimized, stateless music recommendation backend built with Django REST Framework (DRF). Unlike traditional Deep Learning or Collaborative Filtering systems, NextTrack leverages **Content-Based Filtering (CBF)** using pure acoustic features (Energy, Valence, Danceability, Acousticness) through a high-dimensional vector search engine (Annoy). 

It operates entirely in-memory using a dataset of 1.16M tracks, utilizing a brilliant **"One-Way Cold Start Injection"** architecture to allow users to search for brand new global releases and use them instantly as contextual anchors to discover similar tracks from the local dataset.

---

## ✨ Core Features

* **Sub-Millisecond Vector Search**: Uses Spotify's `Annoy` library for Approximate Nearest Neighbors (ANN) indexing over 1.16M tracks.
* **Hybrid Genre-Aware Reranking**: Combines audio-based geometric distance with categorical metadata (Genre) by applying strict bonus adjustments, improving Precision and MAP scores significantly off the Random Baseline.
* **Customizable Audio Weights**: Allows users to actively shift the mathematical weight multiplier of specific audio features (e.g., boosting *Danceability* by 2.0x) on the fly.
* **Dynamic Cold Start Injection**: Uses the **Soundcharts API** to scrape and inject unindexed, brand new global tracks into RAM dynamically. These tracks immediately function as input contexts without requiring complex index rebuilding.
* **Enterprise API Security**: Integrates strict DRF `Serializers` to manage JSON validation, alongside a bespoke dual-layer Throttling mechanism (1.5-second strict cooldown & 10 requests/minute quota limit).

---

## 🚀 Setup & Installation

### 1. Prerequisites & Dataset Setup
Ensure you have Python 3.10+ installed. 

Because the 1.16M track dataset is extremely large, it has been compressed into a `.zip` file for GitHub storage. **Before running the project, you MUST unpack the dataset**:
1. Navigate to the `data/` folder in the project root.
2. Unzip the dataset file located there.
3. Ensure the extracted file is named **`spotify_data.csv`** and remains directly inside the `data/` folder.

### 2. Environment Setup
Clone the repository and spin up a virtual environment:
```bash
git clone https://github.com/Shinyleaf798/NextTrack_Music_Recommend_API.git
cd NextTrack_Music_Recommend_API

# Create and activate virtual environment (Windows)
python -m venv env
env\Scripts\activate
# (For Mac/Linux: source env/bin/activate)

# Install dependencies
pip install -r requirements.txt
```

### 3. API Keys & Secrets
Never commit real API keys! An `.env.example` has been provided. Create a new file named `.env` locally:
```bash
cp .env.example .env
```
Open `.env` and fill in your real credentials for:
* Django Secret Key
* Spotify Developer Credentials (`SPOTIPY_CLIENT_ID`, `SPOTIPY_CLIENT_SECRET`)
* Soundcharts API Credentials (`SOUNDCHARTS_APP_ID`, `SOUNDCHARTS_API_KEY`)

### 4. Run the Server
Because the dataset loads into Memory and establishes the Annoy Index tree on initialization, booting up the server might take a few seconds.
```bash
python manage.py migrate
python manage.py runserver
```
Visit `http://127.0.0.1:8000/` to test the internal UI Prototype.

---

## 📡 API Endpoints

NextTrack's API is documented and fully interactive via **Swagger UI (drf-spectacular)**. 
Once the server is running, navigate to:
👉 **[http://127.0.0.1:8000/api/docs/](http://127.0.0.1:8000/api/docs/)**

* **`GET /api/tracks/`**
  * Fetches the paginated 1.16M tracks from local memory. Supports `?q=` queries for text search.
* **`POST /api/recommend/`**
  * The core engine. Accepts an array of 1 to 5 `selected_ids`, processes exponential temporal decay across the history, scales user feature weights, and runs the Hybrid Annoy retrieval to return 5 recommended tracks.
* **`GET /api/v0/tracks/{track_id}/`**
  * Hydrates Spotify embed URLs and precise track details via the Spotipy library, falling back to local CSV metadata automatically on failure.
* **`GET /api/external_search/`**
  * Searches the Soundcharts Global Database. Triggers the Cold-Start injection logic while maintaining strict DRF dual-throttle protection.

---

## 🧪 Testing & Benchmarks

The project is packaged with a custom Offline Evaluator that runs isolated benchmark environments.
```bash
# Run unit tests and API validations
python manage.py test

# Run the Offline Evaluation Benchmark (Precision@5, Recall@5, Novelty, Diversity, MAP)
python tests_evaluation/run_benchmark_100.py
```