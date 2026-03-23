import sys
import os
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from NextTrack_back.utils.data_loader import data_store
from NextTrack_back.services.recommender import recommender
from NextTrack_back.utils.evaluator import OfflineEvaluator

# Seed for reproducibility (100 scenarios)
RANDOM_SEED = 24

# Relevance criteria: genre match AND cosine similarity > threshold
RELEVANCE_THRESHOLD = 0.80

# Feature weights configuration for Mode B testing
FEATURE_WEIGHTS_MODE_B = {
    'energy':       2.0,
    'danceability': 2.0,
    'valence':      1.0,
    'acousticness': 0.5,
}

def _empty_metrics():
    return {"precision": [], "recall": [], "map": [], "novelty": [], "diversity": []}

def _get_feature_vectors(result_list):
    # Returns feature matrix preserving the original ranking order
    ids = [r.get('track_id') for r in result_list if r.get('track_id')]
    if not ids:
        return np.zeros((1, data_store.feature_matrix.shape[1]))
        
    features = data_store.get_features_by_ids(ids)
    if features is None or len(features) == 0:
        return np.zeros((1, data_store.feature_matrix.shape[1]))
        
    return features

def run_single_mode(mode_name, user_weights=None, n=100):
    print(f"\n========================================================================")
    print(f"  {mode_name}")
    print(f"  Criteria: Genre match AND Cosine Similarity > {RELEVANCE_THRESHOLD}")
    print(f"  Baselines: Random, Popularity")
    print(f"========================================================================\n")

    summary_data = {
        "KNN Model":           _empty_metrics(),
        "Random Baseline":     _empty_metrics(),
        "Popularity Baseline": _empty_metrics(),
    }

    # Pre-calculate popular tracks pool to draw from
    top_20_popular = data_store.df.nlargest(20, 'popularity')

    for i in range(1, n + 1):
        # Sample 3 random tracks to simulate user history
        sample_input = data_store.df.sample(3, random_state=RANDOM_SEED + i).reset_index(drop=True)
        input_ids    = sample_input['track_id'].tolist()

        # Target ground truth
        target_genre = sample_input.iloc[-1]['genre']

        # Build weighted target vector (exponential time-decay)
        N = len(input_ids)
        lam = 0.1
        decay_weights = np.array([np.exp(-lam * (N - j - 1)) for j in range(N)])
        decay_weights /= decay_weights.sum()
        input_vectors = data_store.get_features_by_ids(input_ids)
        if input_vectors is None or len(input_vectors) == 0:
            continue
        target_vector = np.average(input_vectors, axis=0, weights=decay_weights).reshape(1, -1)

        # Apply custom mode weights if provided
        if user_weights:
            W = np.ones((1, len(data_store.feature_cols)))
            feature_mapping = {
                'energy':       data_store.feature_cols.index('energy'),
                'valence':      data_store.feature_cols.index('valence'),
                'danceability': data_store.feature_cols.index('danceability'),
                'acousticness': data_store.feature_cols.index('acousticness'),
            }
            for f_name, col_idx in feature_mapping.items():
                if f_name in user_weights:
                    W[0, col_idx] = float(user_weights[f_name])
            weighted_target = target_vector * W
        else:
            W = None
            weighted_target = target_vector

        # Retrieve predictions
        knn_res, _  = recommender.recommend(input_ids, user_weights=user_weights)
        random_res  = data_store.df[~data_store.df['track_id'].isin(input_ids)].sample(5, random_state=RANDOM_SEED + i).to_dict('records')
        pop_res     = top_20_popular[~top_20_popular['track_id'].isin(input_ids)].head(5).to_dict('records')

        engines = [
            ("KNN Model",           knn_res),
            ("Random Baseline",     random_res),
            ("Popularity Baseline", pop_res),
        ]

        # Calculate metrics for each engine
        for name, results in engines:
            if not results:
                continue

            genres     = [r.get('genre', '')      for r in results]
            pops       = [r.get('popularity', 50) for r in results]
            rec_vecs   = _get_feature_vectors(results)

            # Annoy recall pool (150 candidates)
            pool_indices, _ = recommender.annoy_index.get_nns_by_vector(
                target_vector[0], 150, include_distances=True
            )
            pool_vecs    = data_store.feature_matrix[pool_indices]
            pool_genres  = data_store.df.iloc[pool_indices]['genre'].tolist()

            # Align evaluation vectors for weighted modes
            if user_weights and W is not None:
                eval_target    = weighted_target
                eval_rec_vecs  = rec_vecs * W
                eval_pool_vecs = pool_vecs * W
            else:
                eval_target    = target_vector
                eval_rec_vecs  = rec_vecs
                eval_pool_vecs = pool_vecs

            summary_data[name]["precision"].append(
                OfflineEvaluator.get_combined_precision(
                    genres, eval_rec_vecs, target_genre, eval_target, RELEVANCE_THRESHOLD
                )
            )
            summary_data[name]["recall"].append(
                OfflineEvaluator.get_combined_recall(
                    genres, eval_rec_vecs,
                    pool_genres, eval_pool_vecs,
                    target_genre, eval_target, RELEVANCE_THRESHOLD
                )
            )
            summary_data[name]["map"].append(
                OfflineEvaluator.get_combined_map(
                    genres, eval_rec_vecs, target_genre, eval_target, RELEVANCE_THRESHOLD
                )
            )
            summary_data[name]["novelty"].append(OfflineEvaluator.get_novelty(pops))
            summary_data[name]["diversity"].append(OfflineEvaluator.get_diversity(rec_vecs))

        if i % 20 == 0:
            print(f"  Completed {i}/{n} scenarios...")

    # Output results
    print(f"\n--------------------------------------------------------------------------------")
    print(f"{'Strategy':<25} | {'Prec@5':>7} | {'Recall@5':>8} | {'MAP':>7} | {'Novelty':>8} | {'Diversity':>9}")
    print(f"--------------------------------------------------------------------------------")
    for name, metrics in summary_data.items():
        p   = np.mean(metrics["precision"])
        r   = np.mean(metrics["recall"])
        m   = np.mean(metrics["map"])
        nov = np.mean(metrics["novelty"])
        div = np.mean(metrics["diversity"])
        print(f"{name:<25} | {p:>7.4f} | {r:>8.4f} | {m:>7.4f} | {nov:>8.2f} | {div:>9.4f}")
    print(f"--------------------------------------------------------------------------------")

    return summary_data


def run_benchmark():
    print("  NextTrack Offline Evaluation - Dual Mode Benchmark (100 Scenarios each)")

    results_a = run_single_mode(
        mode_name    = "MODE A - Default KNN (No Feature Weighting)",
        user_weights = None,
        n            = 100,
    )

    results_b = run_single_mode(
        mode_name    = "MODE B - KNN + Feature Weighting",
        user_weights = FEATURE_WEIGHTS_MODE_B,
        n            = 100,
    )

    print("\n========================================================================")
    print("  COMPARISON: Mode A (Default) vs Mode B (Weighted)")
    print("========================================================================")
    print(f"{'Metric':<12} | {'Mode A (Default)':>18} | {'Mode B (Weighted)':>18} | {'Improvement':>14}")
    print(f"------------------------------------------------------------------------")
    for key, label in [
        ("precision", "Precision@5"),
        ("recall",    "Recall@5"),
        ("map",       "MAP"),
        ("novelty",   "Novelty"),
        ("diversity", "Diversity"),
    ]:
        a_val = np.mean(results_a["KNN Model"][key])
        b_val = np.mean(results_b["KNN Model"][key])
        delta = b_val - a_val
        arrow = "Up" if delta > 0 else ("Down" if delta < 0 else "─")
        print(f"{label:<12} | {a_val:>18.4f} | {b_val:>18.4f} | {arrow} {abs(delta):>11.4f}")
    print("========================================================================")


if __name__ == "__main__":
    run_benchmark()
