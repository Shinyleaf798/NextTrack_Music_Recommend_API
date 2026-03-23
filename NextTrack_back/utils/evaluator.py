import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class OfflineEvaluator:
    # --- Genre-Based Metrics ---
    
    @staticmethod
    def get_precision(recommended_genres, target_genre):
        matches = [g for g in recommended_genres if g == target_genre]
        return len(matches) / len(recommended_genres) if recommended_genres else 0

    @staticmethod
    def get_recall(recommended_genres, target_genre, total_relevant_in_dataset):
        matches = sum(1 for g in recommended_genres if g == target_genre)
        return matches / total_relevant_in_dataset if total_relevant_in_dataset > 0 else 0

    @staticmethod
    def get_map(recommended_genres, target_genre):
        hits = 0
        sum_precision = 0
        for i, genre in enumerate(recommended_genres, 1):
            if genre == target_genre:
                hits += 1
                sum_precision += hits / i
        return sum_precision / hits if hits > 0 else 0


    # --- Combined Metrics (Genre + Cosine Similarity > threshold) ---
    
    @staticmethod
    def get_combined_precision(recommended_genres, recommended_vectors,
                               target_genre, target_vector, threshold=0.80):
        if not recommended_genres or recommended_vectors is None or len(recommended_vectors) == 0:
            return 0.0
        sims = cosine_similarity(target_vector, recommended_vectors)[0]
        relevant = sum(
            1 for g, s in zip(recommended_genres, sims)
            if g == target_genre and s > threshold
        )
        return float(relevant) / len(recommended_genres)

    @staticmethod
    def get_combined_recall(recommended_genres, recommended_vectors,
                            all_candidate_genres, all_candidate_vectors,
                            target_genre, target_vector, threshold=0.80):
        if not recommended_genres or recommended_vectors is None:
            return 0.0
        pool_sims = cosine_similarity(target_vector, all_candidate_vectors)[0]
        total_relevant = sum(
            1 for g, s in zip(all_candidate_genres, pool_sims)
            if g == target_genre and s > threshold
        )
        if total_relevant == 0:
            return 0.0
        rec_sims = cosine_similarity(target_vector, recommended_vectors)[0]
        retrieved = sum(
            1 for g, s in zip(recommended_genres, rec_sims)
            if g == target_genre and s > threshold
        )
        return float(retrieved) / total_relevant

    @staticmethod
    def get_combined_map(recommended_genres, recommended_vectors,
                         target_genre, target_vector, threshold=0.80):
        if not recommended_genres or recommended_vectors is None or len(recommended_vectors) == 0:
            return 0.0
        sims = cosine_similarity(target_vector, recommended_vectors)[0]
        hits = 0
        sum_precision = 0.0
        for i, (genre, sim) in enumerate(zip(recommended_genres, sims), 1):
            if genre == target_genre and sim > threshold:
                hits += 1
                sum_precision += hits / i
        return sum_precision / hits if hits > 0 else 0.0


    # --- Novelty & Diversity ---

    @staticmethod
    def get_novelty(recommended_popularities):
        return np.mean([100 - p for p in recommended_popularities])

    @staticmethod
    def get_diversity(feature_matrix):
        if len(feature_matrix) < 2:
            return 0
        sim = cosine_similarity(feature_matrix)
        avg_sim = (np.sum(sim) - len(feature_matrix)) / (len(feature_matrix) * (len(feature_matrix) - 1))
        return 1 - avg_sim
