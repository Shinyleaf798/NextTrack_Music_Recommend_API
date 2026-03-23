from django.urls import path
from . import views

urlpatterns = [
    # HTML page
    path('', views.index, name='index'),

    # API 1: List & search tracks
    path('api/tracks/', views.list_tracks_api, name='list_tracks'),

    # API 2: Get recommendations
    path('api/recommend/', views.recommendations_api, name='recommendations_api'),

    # API 3: Single track detail (via Spotify API)
    path('api/v0/tracks/<str:track_id>/', views.track_detail_api, name='track_detail'),
    
    # API 4: Soundcharts external fallback search
    path('api/external_search/', views.external_search_api, name='external_search_api'),
]