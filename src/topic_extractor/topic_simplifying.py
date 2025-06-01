"""
Semantic similarity-based topic mapping to hierarchical taxonomy using word embeddings.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Set, Tuple, Union, Any, Optional
from gensim.models import KeyedVectors
from gensim import downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicTaxonomyMapper:
    """Maps topic words to hierarchical taxonomy using word embeddings."""

    def __init__(self, model_name: str = "word2vec-google-news-300"):
        """Initialize with pre-trained word embedding model."""
        try:
            logger.info(f"Loading word embedding model: {model_name}")
            self.model: KeyedVectors = api.load(model_name)
            self.embedding_dim = self.model.vector_size
            logger.info(f"Loaded model with {self.embedding_dim}-dimensional vectors")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            logger.info("Falling back to smaller GloVe model...")
            try:
                self.model: KeyedVectors = api.load("glove-wiki-gigaword-50")
                self.embedding_dim = self.model.vector_size
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback model: {fallback_error}")
                raise RuntimeError(
                    "Could not load any word embedding model"
                ) from fallback_error

        self.taxonomy = self._build_taxonomy()
        self._precompute_embeddings()
        logger.info(
            f"Initialized taxonomy mapper with {len(self.taxonomy)} major topics"
        )

    def _build_taxonomy(self) -> Dict[str, List[str]]:
        """Build comprehensive taxonomy covering major domains of human discourse."""
        taxonomy = {
            "relationships": [
                "romantic_relationships",
                "dating",
                "marriage",
                "breakups",
                "crushes",
                "friendship",
                "family_relationships",
                "social_connections",
                "love",
                "intimacy",
                "partnership",
                "trust_issues",
            ],
            "religion": [
                "all_religions",
                "agnosticism",
                "atheism",
                "spiritual_experiences",
                "spiritual_growth",
                "spiritual_practices",
                "spiritual_community",
            ],
            "politics": [
                "political_commentary",
                "political_news",
                "political_analysis",
                "republican",
                "democrat",
                "conservative",
                "liberal",
                "socialism",
                "capitalism",
                "social_justice",
            ],
            "self-improvement": [
                "goal_setting",
                "motivation",
                "confidence_building",
                "life_changes",
                "personal_growth",
                "mindfulness",
                "self_reflection",
                "habits",
                "productivity",
                "time_management",
                "self_care",
                "identity",
            ],
            "emotions": [
                "happiness",
                "sadness",
                "anger",
                "anxiety",
                "depression",
                "stress",
                "emotional_wellbeing",
                "mental_health",
                "therapy",
                "mood_changes",
                "emotional_expression",
                "coping_strategies",
            ],
            "entertainment": [
                "movies",
                "television",
                "music",
                "concerts",
                "theater",
                "comedy",
                "celebrities",
                "entertainment_news",
                "pop_culture",
                "streaming",
                "festivals",
                "awards_shows",
                "media_content",
            ],
            "technology": [
                "computers",
                "smartphones",
                "internet",
                "social_media",
                "apps",
                "software",
                "gadgets",
                "artificial_intelligence",
                "programming",
                "digital_trends",
                "tech_reviews",
                "cybersecurity",
                "innovation",
            ],
            "work": [
                "career",
                "workplace_issues",
                "professional_skills",
                "entrepreneurship",
                "business",
                "management",
                "leadership",
                "workplace_relationships",
            ],
            "education": [
                "school",
                "university",
                "studying",
                "academic_achievement",
                "learning_skills",
                "educational_experiences",
                "teachers",
                "courses",
                "research",
                "knowledge",
                "skill_development",
            ],
            "health": [
                "physical_health",
                "exercise",
                "nutrition",
                "diet",
                "medical_issues",
                "wellness_practices",
                "fitness",
                "healthcare",
                "body_image",
                "lifestyle",
                "preventive_care",
                "health_goals",
            ],
            "travel": [
                "vacation",
                "destinations",
                "travel_experiences",
                "cultural_exploration",
                "adventure",
                "tourism",
                "travel_planning",
                "international_travel",
                "local_exploration",
                "transportation",
                "accommodations",
            ],
            "food": [
                "recipes",
                "cooking",
                "restaurants",
                "cuisine",
                "food_culture",
                "baking",
                "nutrition",
                "food_reviews",
                "dining_experiences",
                "beverages",
                "food_preparation",
                "culinary_skills",
            ],
            "sports": [
                "football",
                "basketball",
                "soccer",
                "tennis",
                "baseball",
                "athletics",
                "sports_events",
                "team_sports",
                "individual_sports",
                "sports_news",
                "competitive_sports",
                "training",
                "running",
                "cycling",
                "swimming",
                "golf",
            ],
            "arts_culture": [
                "visual_arts",
                "literature",
                "poetry",
                "creative_writing",
                "photography",
                "painting",
                "sculpture",
                "cultural_events",
                "artistic_expression",
                "museums",
                "galleries",
                "cultural_heritage",
            ],
            "family": [
                "parenting",
                "children",
                "family_life",
                "family_events",
                "siblings",
                "extended_family",
                "family_traditions",
                "family_relationships",
                "family_activities",
                "family_support",
                "family_dynamics",
            ],
            "hobbies_interests": [
                "collecting",
                "crafting",
                "gardening",
                "reading",
                "gaming",
                "music_making",
                "outdoor_activities",
                "creative_hobbies",
                "recreational_pursuits",
                "hobby_communities",
                "leisure_activities",
            ],
            "finance_money": [
                "personal_finance",
                "budgeting",
                "savings",
                "investments",
                "debt",
                "financial_planning",
                "money_management",
                "economic_issues",
                "financial_advice",
                "spending",
                "financial_goals",
            ],
            "home": [
                "home_improvement",
                "interior_design",
                "household_management",
                "home_maintenance",
                "living_spaces",
                "home_decoration",
                "organization",
                "domestic_life",
                "housing",
                "neighborhood",
            ],
            "fashion": [
                "clothing",
                "fashion_trends",
                "personal_style",
                "beauty",
                "makeup",
                "fashion_advice",
                "style_inspiration",
                "accessories",
                "grooming",
                "appearance",
            ],
            "nature": [
                "environmental_issues",
                "climate_change",
                "nature_appreciation",
                "outdoor_experiences",
                "wildlife",
                "conservation",
                "sustainability",
                "environmental_activism",
                "ecological_awareness",
            ],
            "news": [
                "current_events",
                "political_news",
                "world_events",
                "local_news",
                "breaking_news",
                "current_affairs",
                "news_analysis",
                "social_issues",
                "public_policy",
                "global_events",
                "journalism",
                "media_coverage",
            ],
            "communication": [
                "social_interactions",
                "communication_skills",
                "social_media_use",
                "online_communities",
                "networking",
                "social_trends",
                "digital_communication",
                "social_behavior",
                "community_involvement",
            ],
            "transportation": [
                "driving",
                "public_transportation",
                "vehicles",
                "traffic",
                "commuting",
                "transportation_issues",
                "automotive",
                "travel_methods",
                "mobility",
            ],
            "science": [
                "scientific_discoveries",
                "research",
                "scientific_method",
                "biology",
                "physics",
                "chemistry",
                "space",
                "scientific_innovation",
                "laboratories",
                "scientific_studies",
            ],
            "history": [
                "historical_events",
                "historical_figures",
                "cultural_history",
                "historical_analysis",
                "historical_periods",
                "heritage",
                "historical_significance",
                "historical_research",
            ],
            "gaming": [
                "video_games",
                "gaming_culture",
                "game_reviews",
                "gaming_technology",
                "competitive_gaming",
                "game_development",
                "gaming_communities",
                "mobile_games",
                "gaming_industry",
                "game_design",
            ],
            "music": [
                "musical_genres",
                "musicians",
                "concerts",
                "music_creation",
                "instruments",
                "music_appreciation",
                "music_industry",
                "songwriting",
                "music_performance",
                "music_technology",
                "music_education",
            ],
            "literature": [
                "books",
                "authors",
                "creative_writing",
                "literary_analysis",
                "poetry",
                "storytelling",
                "writing_process",
                "literary_genres",
                "publishing",
                "reading_experiences",
                "literary_criticism",
            ],
            "photography": [
                "photography_techniques",
                "visual_storytelling",
                "photo_editing",
                "photography_equipment",
                "visual_arts",
                "image_composition",
                "photography_genres",
                "visual_media",
                "graphic_design",
            ],
            "social_issues": [
                "social_justice",
                "community_involvement",
                "volunteer_work",
                "activism",
                "social_change",
                "community_development",
                "civic_engagement",
                "social_responsibility",
                "community_events",
            ],
            "lifestyle": [
                "daily_routines",
                "life_choices",
                "drug_use",
                "alcohol_use",
                "smoking",
                "personal_preferences",
                "lifestyle_trends",
                "quality_of_life",
                "life_experiences",
                "lifestyle_balance",
            ],
        }

        logger.info(
            f"Built taxonomy with {len(taxonomy)} major topics and "
            f"{sum(len(subtopics) for subtopics in taxonomy.values())} subtopics"
        )
        return taxonomy

    def _get_word_embedding(self, word: str) -> Optional[np.ndarray]:
        """Get embedding vector for a single word."""
        try:
            # Try normalized word first
            normalized_word = word.lower().replace("_", " ")
            if normalized_word in self.model.key_to_index:
                return self.model[normalized_word]

            # Try original word
            if word in self.model.key_to_index:
                return self.model[word]

            # Try with underscores replaced by spaces
            space_word = word.replace("_", " ")
            if space_word in self.model.key_to_index:
                return self.model[space_word]

            return None
        except (KeyError, Exception):
            return None

    def _get_phrase_embedding(self, phrase: str) -> Optional[np.ndarray]:
        """Get embedding for a phrase by averaging word embeddings."""
        words = phrase.replace("_", " ").lower().split()
        word_embeddings = []

        for word in words:
            embedding = self._get_word_embedding(word)
            if embedding is not None:
                word_embeddings.append(embedding)

        if not word_embeddings:
            return None

        return np.mean(word_embeddings, axis=0)

    def _precompute_embeddings(self) -> None:
        """Precompute embeddings for all taxonomy terms."""
        self.subtopic_embeddings = {}
        self.subtopic_to_major = {}

        logger.info("Computing embeddings for taxonomy terms...")

        for major_topic, subtopics in self.taxonomy.items():
            for subtopic in subtopics:
                embedding = self._get_phrase_embedding(subtopic)
                if embedding is not None:
                    self.subtopic_embeddings[subtopic] = embedding
                    self.subtopic_to_major[subtopic] = major_topic
                else:
                    logger.warning(
                        f"Could not compute embedding for subtopic: {subtopic}"
                    )

        logger.info(
            f"Successfully computed embeddings for {len(self.subtopic_embeddings)} subtopics"
        )

    def _compute_semantic_similarity(
        self, input_embedding: np.ndarray, target_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Compute cosine similarity between input and target embeddings."""
        similarities = {}

        for category, target_embedding in target_embeddings.items():
            input_reshaped = input_embedding.reshape(1, -1)
            target_reshaped = target_embedding.reshape(1, -1)
            similarity = cosine_similarity(input_reshaped, target_reshaped)[0][0]
            similarities[category] = float(similarity)

        return similarities


def map_lda_results_to_taxonomy(
    mapper: TopicTaxonomyMapper,
    lda_results: Dict[str, Any],
    top_n: int = 10,
    min_similarity: float = 0.20,
) -> Dict[str, float]:
    """
    Map LDA analysis results to taxonomy categories using weighted word embeddings.

    Args:
        mapper: TopicTaxonomyMapper instance
        lda_results: Dictionary containing LDA analysis results
        top_n: Number of top taxonomy matches to return
        min_similarity: Minimum similarity threshold (as decimal, e.g., 0.20 = 20%)

    Returns:
        Dictionary mapping "major:subtopic" to percentage similarity scores
    """
    if not lda_results or "lda_results" not in lda_results:
        logger.error("Invalid LDA results structure")
        return {}

    lda_data = lda_results["lda_results"]

    # Use high quality topics if available, otherwise all topics
    if "high_quality_topics" in lda_data and lda_data["high_quality_topics"]:
        topics_to_analyze = lda_data["high_quality_topics"]
        logger.info(f"Using {len(topics_to_analyze)} high quality topics")
    elif "topics" in lda_data and lda_data["topics"]:
        topics_to_analyze = lda_data["topics"]
        logger.info(f"Using all {len(topics_to_analyze)} topics")
    else:
        logger.error("No topics found in LDA results")
        return {}

    all_results = {}

    for topic in topics_to_analyze:
        topic_id = topic.get("topic_id", "unknown")
        words = topic.get("words", [])
        weights = topic.get("weights", [])
        coherence_scores = topic.get("coherence_scores", [])
        topic_quality = topic.get("topic_quality", 0.5)

        if not words or not weights or len(words) != len(weights):
            logger.warning(f"Topic {topic_id} has mismatched words/weights, skipping")
            continue

        # Use coherence scores if available, otherwise use default values
        if not coherence_scores or len(coherence_scores) != len(words):
            coherence_scores = [0.5] * len(words)

        # Select top words by weight (at least top 3, max 8)
        word_data = list(zip(words, weights, coherence_scores))
        word_data.sort(key=lambda x: x[1], reverse=True)

        # Take top 60th percentile or minimum 3 words, max 8
        threshold = np.percentile(weights, 60) if len(weights) > 3 else min(weights)
        selected_data = [item for item in word_data if item[1] >= threshold][:8]

        if len(selected_data) < 3:
            selected_data = word_data[:3]

        selected_words = [item[0] for item in selected_data]
        selected_weights = [item[1] for item in selected_data]
        selected_coherence = [item[2] for item in selected_data]

        logger.info(f"Topic {topic_id}: Using {len(selected_words)} words")

        # Create weighted topic embedding
        topic_embedding = _create_weighted_embedding(
            selected_words, selected_weights, selected_coherence, mapper
        )

        if topic_embedding is None:
            logger.warning(f"Topic {topic_id}: No valid embeddings found")
            continue

        # Compute similarities with subtopics
        subtopic_similarities = mapper._compute_semantic_similarity(
            topic_embedding, mapper.subtopic_embeddings
        )

        # Apply quality weighting and convert to results format
        quality_multiplier = 0.5 + (topic_quality * 0.5)  # Scale to 0.5-1.0

        for subtopic, similarity in subtopic_similarities.items():
            weighted_similarity = similarity * quality_multiplier

            if weighted_similarity >= min_similarity:
                major_topic = mapper.subtopic_to_major[subtopic]
                key = f"{major_topic}:{subtopic}"
                percentage = round(weighted_similarity * 100, 2)

                # Keep highest score if duplicate keys exist
                if key not in all_results or percentage > all_results[key]:
                    all_results[key] = percentage

    # Return top results sorted by score
    sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
    final_results = dict(sorted_results[: top_n * 2])

    logger.info(f"Mapped topics to {len(final_results)} taxonomy categories")
    return final_results


def _create_weighted_embedding(
    words: List[str],
    weights: List[float],
    coherence_scores: List[float],
    mapper: TopicTaxonomyMapper,
) -> Optional[np.ndarray]:
    """Create weighted topic embedding combining LDA weights and coherence scores."""
    embeddings = []
    final_weights = []

    for word, weight, coherence in zip(words, weights, coherence_scores):
        embedding = mapper._get_phrase_embedding(word)
        if embedding is not None:
            # Combine LDA weight with coherence score
            combined_weight = weight * (
                coherence + 0.2
            )  # Add 0.2 to avoid zero multiplication
            embeddings.append(embedding)
            final_weights.append(combined_weight)

    if not embeddings:
        return None

    # Normalize weights and compute weighted average
    final_weights = np.array(final_weights)
    final_weights = final_weights / np.sum(final_weights)

    return np.average(embeddings, axis=0, weights=final_weights)
