import os
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


# Singleton pattern to ensure only one instance loads the dataset into RAM
class DataLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        self.file_path = os.path.join(BASE_DIR, 'data', 'spotify_data.csv')

        self.feature_cols = [
            'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo'
        ]

        self.df = None
        self.feature_matrix = None
        self.scaler = MinMaxScaler()

        self.load_data()
        self._initialized = True

    # Load CSV into Pandas DataFrame and pre-compute scaled feature matrix
    def load_data(self):
        start_time = time.time()
        try:
            raw_df = pd.read_csv(self.file_path)

            # Drop duplicates to maintain a clean lookup table
            self.df = raw_df.drop_duplicates(subset='track_id').reset_index(drop=True)

            # Normalize all features to [0, 1] range for KNN cosine similarity
            numeric_data = self.df[self.feature_cols]
            self.feature_matrix = self.scaler.fit_transform(numeric_data)

            duration = time.time() - start_time
            logger.info(f"Loaded {len(self.df)} tracks into RAM in {duration:.4f}s.")

        except FileNotFoundError:
            logger.error(f"Dataset not found at {self.file_path}. Check the data directory.")
        except Exception as e:
            logger.error(f"Unexpected error during data loading: {e}")

    # Retrieve full track metadata row by ID
    def get_track_by_id(self, track_id):
        if self.df is None:
            return None
        result = self.df[self.df['track_id'] == track_id]
        return result.iloc[0] if not result.empty else None

    # Return a numpy feature matrix for the given IDs, preserving input order
    def get_features_by_ids(self, track_ids):
        if self.df is None or self.feature_matrix is None:
            return None
            
        indices = []
        for tid in track_ids:
            match = self.df[self.df['track_id'] == tid].index
            if not match.empty:
                indices.append(match[0])
                
        if not indices:
            return None
        return self.feature_matrix[indices]

    # Append dynamically fetched tracks (Soundcharts) to the in-memory dataset
    def inject_external_track(self, track_dict):
        if self.df is None or self.feature_matrix is None:
            logger.warning("Cannot inject track: data store is not initialised.")
            return

        track_id = track_dict.get('track_id')
        if not track_id:
            logger.warning("Cannot inject track: missing track_id.")
            return

        # Skip if already in memory
        if track_id in self.df['track_id'].values:
            return

        new_df = pd.DataFrame([track_dict])

        # Fill missing features with 0.0 to prevent scaling errors
        for col in self.feature_cols:
            if col not in new_df.columns:
                new_df[col] = 0.0

        self.df = pd.concat([self.df, new_df], ignore_index=True)

        # Scale new track using the exact same scaler fit on the original dataset
        feature_data = new_df[self.feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        scaled_new = self.scaler.transform(feature_data)
        self.feature_matrix = np.vstack([self.feature_matrix, scaled_new])

        logger.info(f"Injected external track: {track_dict.get('track_name', 'unknown')}")


# Global instance imported by views/services
data_store = DataLoader()