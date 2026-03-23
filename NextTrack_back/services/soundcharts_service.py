import os
import requests
import json
import logging
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(os.path.join(BASE_DIR, '.env'))

logger = logging.getLogger(__name__)


class SoundchartsService:
    def __init__(self):
        self.app_id       = os.getenv('SOUNDCHARTS_APP_ID')
        self.api_key      = os.getenv('SOUNDCHARTS_API_KEY')
        self.base_url_v2  = "https://customer.api.soundcharts.com/api/v2"
        self.base_url_v2_25 = "https://customer.api.soundcharts.com/api/v2.25"
        
        self.headers = {
            "x-app-id":  self.app_id,
            "x-api-key": self.api_key
        }

    def _is_configured(self):
        return bool(self.app_id and self.api_key)

    def search_and_fetch_track(self, query):
        if not self._is_configured():
            logger.error("Soundcharts API credentials missing.")
            return None, "Soundcharts API credentials missing in .env"

        # 1. Search for a track by name -> get UUID (Top 1)
        search_url = f"{self.base_url_v2}/song/search/{query}"
        try:
            search_res = requests.get(search_url, headers=self.headers, params={"limit": 5}, timeout=5)
            search_res.raise_for_status()
            search_data = search_res.json()
            
            items = search_data.get('items', [])
            if not items:
                return None, f"No external results found for '{query}'."
            
            track_uuid = items[0].get('uuid')
            
        except Exception as e:
            logger.error(f"Soundcharts search failed: {e}")
            return None, "Failed to connect to global library search."

        # 2. Fetch Deep Metadata containing Audio Features (v2.25)
        meta_url = f"{self.base_url_v2_25}/song/{track_uuid}"
        try:
            meta_res = requests.get(meta_url, headers=self.headers, timeout=5)
            meta_res.raise_for_status()
            meta_data = meta_res.json()
        except Exception as e:
            logger.error(f"Soundcharts deep metadata fetch failed: {e}")
            return None, "Failed to retrieve track details from global library."

        # 3. Parse, prune, and standardize the dict to match local dataset schema
        try:
            meta_data = meta_data.get('object', meta_data)
            
            track_name = meta_data.get('name', 'Unknown Title')
            
            # Extract root genre
            genres = meta_data.get('genres', [])
            genre = genres[0].get('root', 'unknown') if genres else 'unknown'
            
            # Get performing artist name, prioritizing 'creditName'
            artist_name = meta_data.get('creditName')
            if not artist_name:
                main_artists = meta_data.get('mainArtists', []) or meta_data.get('artists', [])
                if main_artists:
                    artist_names = [a.get('name') for a in main_artists]
                    artist_name = ", ".join(artist_names[:2])
                else:
                    artist_name = "Unknown Artist"

            audio = meta_data.get('audio', {})
            
            normalized_track = {
                'track_id':         track_uuid,
                'track_name':       track_name,
                'artist_name':      artist_name,
                'genre':            genre,
                'danceability':     float(audio.get('danceability', 0.5)),
                'energy':           float(audio.get('energy', 0.5)),
                'key':              float(audio.get('key', 0)),
                'loudness':         float(audio.get('loudness', -10.0)),
                'mode':             float(audio.get('mode', 1)),
                'speechiness':      float(audio.get('speechiness', 0.1)),
                'acousticness':     float(audio.get('acousticness', 0.5)),
                'instrumentalness': float(audio.get('instrumentalness', 0.0)),
                'liveness':         float(audio.get('liveness', 0.2)),
                'valence':          float(audio.get('valence', 0.5)),
                'tempo':            float(audio.get('tempo', 120.0)),
                'source':           'soundcharts'
            }
            return normalized_track, None

        except Exception as e:
            logger.error(f"Data pruning error for Soundcharts response: {e}")
            return None, "Error parsing external track data."


soundcharts_service = SoundchartsService()
