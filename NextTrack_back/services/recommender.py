import os
import math
import numpy as np
from annoy import AnnoyIndex
from scipy.spatial.distance import cdist
from ..utils.data_loader import data_store


class MusicRecommender:
    def __init__(self):
        if data_store.feature_matrix is None:
            self.annoy_index = None
            return

        self.f_dim = data_store.feature_matrix.shape[1]
        self.annoy_index = AnnoyIndex(self.f_dim, 'angular')

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.index_path = os.path.join(current_dir, 'spotify_index.ann')

        if os.path.exists(self.index_path):
            self.annoy_index.load(self.index_path)
        else:
            for i in range(data_store.feature_matrix.shape[0]):
                self.annoy_index.add_item(i, data_store.feature_matrix[i])
            self.annoy_index.build(10)
            self.annoy_index.save(self.index_path)

    def recommend(self, input_track_ids, user_weights=None):
        if self.annoy_index is None:
            return None, "Recommendation index is not available. Please check data loading."

        input_vectors = data_store.get_features_by_ids(input_track_ids)

        if input_vectors is None or len(input_vectors) == 0:
            return None, "Selected tracks are missing from memory. Please re-select them."

        # Exponential time-decay weights: most recent song gets highest importance
        N = len(input_vectors)
        lam = 0.1
        weights = np.array([np.exp(-lam * (N - i - 1)) for i in range(N)])
        weights = weights / weights.sum()

        target_vector = np.average(input_vectors, axis=0, weights=weights).reshape(1, -1)

        # Determine the dominant genre from input tracks for hybrid reranking
        input_rows = data_store.df[data_store.df['track_id'].isin(input_track_ids)]
        dominant_genre = input_rows['genre'].mode()[0] if not input_rows.empty else None

        # Fast Search (Recall) - top 150 candidates
        recall_k = 150
        stage1_indices, stage1_distances = self.annoy_index.get_nns_by_vector(
            target_vector[0], recall_k, include_distances=True
        )
        stage1_distances = np.array([(d**2) / 2 for d in stage1_distances])

        # Precise Ranking
        has_custom_weights = False
        W = np.ones((1, len(data_store.feature_cols)))
        if user_weights:
            feature_mapping = {
                'energy':       data_store.feature_cols.index('energy'),
                'valence':      data_store.feature_cols.index('valence'),
                'danceability': data_store.feature_cols.index('danceability'),
                'acousticness': data_store.feature_cols.index('acousticness'),
            }
            for f_name, col_idx in feature_mapping.items():
                if f_name in user_weights and float(user_weights[f_name]) != 1.0:
                    W[0, col_idx] = float(user_weights[f_name])
                    has_custom_weights = True

        if has_custom_weights:
            candidate_matrix = data_store.feature_matrix[stage1_indices]
            scaled_target = target_vector * W
            scaled_matrix = candidate_matrix * W

            distances_full = cdist(scaled_target, scaled_matrix, metric='cosine')[0]
            local_idx_rank = np.argsort(distances_full)[:20]

            idx_1d = [stage1_indices[i] for i in local_idx_rank]
            distances_1d = distances_full[local_idx_rank]
        else:
            idx_1d = stage1_indices[:20]
            distances_1d = stage1_distances[:20]

        SIMILARITY_THRESHOLD = 0.8  # Cosine Similarity > 0.8 requirement
        candidates = []

        for i in range(len(idx_1d)):
            idx  = idx_1d[i]
            dist = distances_1d[i]

            raw_info = data_store.df.iloc[idx].to_dict()
            sim_score = round(float(1 - dist), 4)

            # Hybrid Genre-Aware Reranking: +0.05 bonus for same genre
            bonus = 0.05 if raw_info.get('genre') == dominant_genre else 0.0
            final_score = round(sim_score + bonus, 4)

            track_info = {
                'artist_name':      raw_info.get('artist_name'),
                'track_name':       raw_info.get('track_name'),
                'track_id':         raw_info.get('track_id'),
                'genre':            raw_info.get('genre'),
                'popularity':       raw_info.get('popularity', 50),
                'similarity_score': final_score,
            }

            # Guard against JSON serialization errors for NaN
            for k, v in track_info.items():
                if isinstance(v, float) and math.isnan(v):
                    track_info[k] = None

            if track_info['track_id'] in input_track_ids:
                continue

            candidates.append(track_info)

        # Re-sort candidates by the newly calculated hybrid score
        candidates.sort(key=lambda x: x['similarity_score'], reverse=True)

        recommendations = []
        fallback_pool   = []

        # Enforce threshold and collect top 5
        for track_info in candidates:
            if (track_info['similarity_score'] is not None and
                    track_info['similarity_score'] > SIMILARITY_THRESHOLD):
                recommendations.append(track_info)
            else:
                fallback_pool.append(track_info)

            if len(recommendations) == 5:
                break

        # Fallback to next best candidates if threshold criteria isn't met
        if len(recommendations) < 5:
            for track_info in fallback_pool:
                if track_info['track_id'] not in [r['track_id'] for r in recommendations]:
                    recommendations.append(track_info)
                if len(recommendations) == 5:
                    break

        return recommendations[:5], None


recommender = MusicRecommender()