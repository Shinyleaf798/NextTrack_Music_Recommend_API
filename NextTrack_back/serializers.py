from rest_framework import serializers

class TrackListQuerySerializer(serializers.Serializer):
    q = serializers.CharField(required=False, allow_blank=True, help_text="Search query for track name or artist")
    page = serializers.IntegerField(default=1, min_value=1, required=False)
    page_size = serializers.IntegerField(default=15, min_value=1, max_value=100, required=False)

class RecommendationRequestSerializer(serializers.Serializer):
    selected_ids = serializers.ListField(
        child=serializers.CharField(max_length=100),
        min_length=1,
        max_length=5,
        help_text="List of Seed Track IDs (maximum 5)",
        error_messages={
            'min_length': 'You must select at least 1 track.',
            'max_length': 'You can select a maximum of 5 tracks.',
            'empty': 'You must select at least 1 track.'
        }
    )
    energy = serializers.FloatField(default=1.0, required=False, min_value=0.0, max_value=2.0, error_messages={'invalid': 'Feature weights must be valid numbers.', 'max_value': 'Weight cannot exceed 2.0', 'min_value': 'Weight cannot be less than 0.0'})
    valence = serializers.FloatField(default=1.0, required=False, min_value=0.0, max_value=2.0, error_messages={'invalid': 'Feature weights must be valid numbers.', 'max_value': 'Weight cannot exceed 2.0', 'min_value': 'Weight cannot be less than 0.0'})
    danceability = serializers.FloatField(default=1.0, required=False, min_value=0.0, max_value=2.0, error_messages={'invalid': 'Feature weights must be valid numbers.', 'max_value': 'Weight cannot exceed 2.0', 'min_value': 'Weight cannot be less than 0.0'})
    acousticness = serializers.FloatField(default=1.0, required=False, min_value=0.0, max_value=2.0, error_messages={'invalid': 'Feature weights must be valid numbers.', 'max_value': 'Weight cannot exceed 2.0', 'min_value': 'Weight cannot be less than 0.0'})

class ExternalSearchQuerySerializer(serializers.Serializer):
    q = serializers.CharField(required=True, trim_whitespace=True, min_length=1, error_messages={
        'required': 'Please provide a search query.',
        'blank': 'Please provide a search query.'
    })
