import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(os.path.join(BASE_DIR, '.env'))


class SpotifyService:
    def __init__(self):
        auth_manager = SpotifyClientCredentials(
            client_id=os.getenv('SPOTIPY_CLIENT_ID'),
            client_secret=os.getenv('SPOTIPY_CLIENT_SECRET')
        )
        self.sp = spotipy.Spotify(auth_manager=auth_manager)

    def get_track(self, track_id: str) -> dict:
        # Fetch metadata and embed info from Spotify API
        track_data = self.sp.track(track_id)

        artists = ', '.join(a['name'] for a in track_data['artists'])
        images = track_data['album']['images']
        album_art = images[0]['url'] if images else None

        return {
            'track_id':    track_id,
            'track_name':  track_data['name'],
            'artist_name': artists,
            'album':       track_data['album']['name'],
            'album_art':   album_art,
            'preview_url': track_data.get('preview_url'),
            'embed_url':   f'https://open.spotify.com/embed/track/{track_id}',
            'spotify_url': track_data['external_urls']['spotify'],
        }


spotify_service = SpotifyService()
