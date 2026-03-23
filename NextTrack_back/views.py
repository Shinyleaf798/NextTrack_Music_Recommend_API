import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

from django.shortcuts import render
from django.core.paginator import Paginator

from rest_framework.decorators import api_view, throttle_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.throttling import AnonRateThrottle
from drf_spectacular.utils import extend_schema

from .utils.data_loader import data_store
from .services.recommender import recommender
from .services.spotify_service import spotify_service
from .services.soundcharts_service import soundcharts_service
from .serializers import TrackListQuerySerializer, RecommendationRequestSerializer, ExternalSearchQuerySerializer

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(os.path.join(BASE_DIR, '.env'))

logger = logging.getLogger(__name__)

#  HTML Page View
def index(request):
    return render(request, 'index.html')


#  API 1 — GET /api/tracks/
@extend_schema(parameters=[TrackListQuerySerializer], responses={200: dict, 503: dict})
@api_view(['GET'])
def list_tracks_api(request):
    # Guard: data failed to load at startup
    if data_store.df is None:
        return Response({'error': 'Track data is not available.'}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

    serializer = TrackListQuerySerializer(data=request.query_params)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    params = serializer.validated_data
    all_songs = data_store.df[['track_id', 'track_name', 'artist_name', 'genre']].to_dict('records')

    query = params.get('q', '')
    if query:
        q_lower = query.lower()
        all_songs = [s for s in all_songs if q_lower in str(s['track_name']).lower()
                     or q_lower in str(s['artist_name']).lower()]

    page_size = params.get('page_size', 15)
    paginator   = Paginator(all_songs, page_size)
    page_number = params.get('page', 1)
    page_obj    = paginator.get_page(page_number)

    return Response({
        'tracks':      list(page_obj.object_list),
        'total':       paginator.count,
        'page':        page_obj.number,
        'total_pages': paginator.num_pages,
    })


#  API 2 — POST /api/recommend/
@extend_schema(request=RecommendationRequestSerializer, responses={200: dict, 400: dict})
@api_view(['POST'])
def recommendations_api(request):
    serializer = RecommendationRequestSerializer(data=request.data)
    if not serializer.is_valid():
        # Flatten error messages slightly to keep consistent client-facing output
        error_msg = list(serializer.errors.values())[0][0] if serializer.errors else "Invalid data"
        return Response({'error': error_msg}, status=status.HTTP_400_BAD_REQUEST)

    data = serializer.validated_data
    selected_ids = data['selected_ids']
    
    user_weights = {
        'energy':       data.get('energy',       1.0),
        'valence':      data.get('valence',      1.0),
        'danceability': data.get('danceability', 1.0),
        'acousticness': data.get('acousticness', 1.0),
    }

    res, err = recommender.recommend(selected_ids, user_weights=user_weights)

    if err:
        return Response({'error': err}, status=status.HTTP_400_BAD_REQUEST)

    recommendations = res if isinstance(res, list) else res.to_dict('records')
    return Response({'recommendations': recommendations})


#  API 3 — GET /api/v0/tracks/{track_id}/
@extend_schema(responses={200: dict, 404: dict})
@api_view(['GET'])
def track_detail_api(request, track_id):
    try:
        track_data = spotify_service.get_track(track_id)
        track_data['source'] = 'spotify'
        return Response(track_data)

    except Exception as spotify_error:
        # Spotify lookup failed — fall back to local CSV metadata
        logger.warning(f"Spotify lookup failed for '{track_id}': {spotify_error}")

        local_track = data_store.get_track_by_id(track_id)

        if local_track is None:
            return Response({'error': f'Track {track_id!r} not found.'}, status=status.HTTP_404_NOT_FOUND)

        local_track = local_track.to_dict() if hasattr(local_track, 'to_dict') else dict(local_track)

        return Response({
            'track_id':    track_id,
            'track_name':  local_track.get('track_name', ''),
            'artist_name': local_track.get('artist_name', ''),
            'album':       None,
            'album_art':   None,
            'preview_url': None,
            'embed_url':   f'https://open.spotify.com/embed/track/{track_id}',
            'spotify_url': f'https://open.spotify.com/track/{track_id}',
            'source':      'local',
        })


from django.core.cache import cache
from rest_framework.throttling import BaseThrottle
import time

class ExternalSearchCooldownThrottle(BaseThrottle):
    # Custom DRF throttle to strictly enforce a 1.5 second cooldown gap between requests
    def allow_request(self, request, view):
        ip = request.META.get('REMOTE_ADDR', '127.0.0.1')
        self.key = f"ext_cooldown_{ip}"
        self.now = time.time()
        
        last_call = cache.get(self.key)
        if last_call is not None and (self.now - last_call) < 1.5:
            self.wait_time = 1.5 - (self.now - last_call)
            return False
            
        cache.set(self.key, self.now, timeout=2)
        return True

    def wait(self):
        return getattr(self, 'wait_time', 1.5)

class ExternalSearchThrottle(AnonRateThrottle):
    # Enforces 10/min rate limit as defined in settings.py
    scope = 'external_search'


#  API 4 — GET /api/external_search/
@extend_schema(parameters=[ExternalSearchQuerySerializer], responses={200: dict, 400: dict, 404: dict, 429: dict})
@api_view(['GET'])
@throttle_classes([ExternalSearchCooldownThrottle, ExternalSearchThrottle])
def external_search_api(request):
    serializer = ExternalSearchQuerySerializer(data=request.query_params)
    if not serializer.is_valid():
        return Response({'error': 'Please provide a search query.'}, status=status.HTTP_400_BAD_REQUEST)

    query = serializer.validated_data['q']

    track_data, error = soundcharts_service.search_and_fetch_track(query)

    if error:
        return Response({'error': error}, status=status.HTTP_404_NOT_FOUND)

    # Inject into in-memory storage so recommender can use it instantly
    data_store.inject_external_track(track_data)

    return Response({'track': track_data})