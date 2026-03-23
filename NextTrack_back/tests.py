import json
from unittest.mock import patch
from django.test import TestCase, Client
from django.urls import reverse

VALID_IDS = ["id_track_001", "id_track_002", "id_track_003"]

# Fake results returned by the mocked recommender 
FAKE_RECOMMENDATIONS = [
    {"track_id": f"rec_{i}", "track_name": f"Song {i}",
     "artist_name": "Artist", "genre": "pop", "similarity_score": 0.92}
    for i in range(1, 6)
]


# Helper for POST requests to /api/recommend/
def _post_recommend(client, payload, content_type="application/json"):
    body = json.dumps(payload) if isinstance(payload, dict) else payload
    return client.post(
        reverse("recommendations_api"),
        data=body,
        content_type=content_type,
    )


class RecommendAPITests(TestCase):

    def setUp(self):
        self.client = Client()

    # Verify valid inputs return an HTTP 200 with recommendations
    @patch("NextTrack_back.views.recommender")
    def test_valid_request_returns_200(self, mock_recommender):
        mock_recommender.recommend.return_value = (FAKE_RECOMMENDATIONS, None)

        payload = {"selected_ids": VALID_IDS}
        response = _post_recommend(self.client, payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("recommendations", body)
        self.assertIsInstance(body["recommendations"], list)

    # Verify empty track selection returns HTTP 400 validation error
    def test_empty_ids_returns_400(self):
        payload = {"selected_ids": []}
        response = _post_recommend(self.client, payload)

        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertIn("error", body)

    # Verify exceeding the 5-track limit returns HTTP 400
    def test_too_many_ids_returns_400(self):
        payload = {"selected_ids": [f"id_{i}" for i in range(6)]}
        response = _post_recommend(self.client, payload)

        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertIn("error", body)

    # Verify missing tracks in both APIs & local DB return HTTP 404
    def test_invalid_track_id_returns_404(self):
        with patch("NextTrack_back.views.spotify_service") as mock_spotify, \
             patch("NextTrack_back.views.data_store") as mock_ds:

            mock_spotify.get_track.side_effect = Exception("Not found")
            mock_ds.get_track_by_id.return_value = None

            response = self.client.get(
                reverse("track_detail", kwargs={"track_id": "NONEXISTENT_ID_XYZ"})
            )

        self.assertEqual(response.status_code, 404)
        body = response.json()
        self.assertIn("error", body)

    # Verify JSON parse errors are handled safely
    def test_malformed_json_returns_400(self):
        response = _post_recommend(self.client, "{ malformed JSON")

        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertTrue("error" in body or "detail" in body)

    # Verify the external search API rate limit triggers HTTP 429
    def test_rate_limit_returns_429(self):
        from django.core.cache import cache

        cache.clear()

        with patch("NextTrack_back.views.soundcharts_service") as mock_sc, \
             patch("NextTrack_back.views.data_store"):
            mock_sc.search_and_fetch_track.return_value = ({"track_name": "Test"}, None)

            # Exhaust quota (10 per min)
            for _ in range(10):
                self.client.get(reverse("external_search_api"), {"q": "test"})

            # Should hit rate limit
            response = self.client.get(reverse("external_search_api"), {"q": "test"})

        cache.clear()

        self.assertEqual(response.status_code, 429)
        body = response.json()
        # DRF uses 'detail' for throttling messages
        self.assertTrue("error" in body or "detail" in body)
