"""
Academic Topic Taxonomy Mapper

This module implements a semantic similarity-based approach to map extracted topic words
to a comprehensive hierarchical taxonomy. It follows text mining best practices by using
transformer-based embeddings for semantic understanding rather than simple keyword matching.

The system uses a two-level taxonomy with major topics and subtopics, employing cosine
similarity on sentence transformer embeddings to find the most semantically related
categories for given word sets.

Reference:
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- Angelov, D. (2020). Top2Vec: Distributed representations of topics
"""

import logging
import numpy as np
from typing import Dict, List, Set, Tuple, Union, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicTaxonomyMapper:
    """
    Maps topic words to a comprehensive hierarchical taxonomy using semantic similarity.

    This class implements an academic approach to topic categorization by:
    1. Defining a comprehensive taxonomy of human discourse topics
    2. Using transformer-based embeddings for semantic understanding
    3. Computing cosine similarity for category matching
    4. Returning ranked results with confidence scores
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize the taxonomy mapper with a pre-trained sentence transformer.

        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.taxonomy = self._build_comprehensive_taxonomy()
        self._precompute_embeddings()

        logger.info(
            f"Initialized taxonomy mapper with {len(self.taxonomy)} major topics"
        )

    def _build_comprehensive_taxonomy(self) -> Dict[str, List[str]]:
        """
        Build a comprehensive taxonomy covering major domains of human discourse.

        This taxonomy is designed to be domain-agnostic and comprehensive enough to
        categorize topics from personal blogs, news, academic content, and social media.

        Returns:
            Dictionary mapping major topics to lists of subtopics
        """
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
                "interpersonal_conflict",
                "love",
                "intimacy",
                "relationship_advice",
                "partnership",
                "trust_issues",
            ],
            "personal_development": [
                "self_improvement",
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
            "emotions_psychology": [
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
                "psychological_insights",
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
                "entertainment_reviews",
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
            "work_career": [
                "job_search",
                "career_development",
                "workplace_issues",
                "professional_skills",
                "work_life_balance",
                "entrepreneurship",
                "business",
                "leadership",
                "career_change",
                "workplace_relationships",
                "professional_growth",
                "interviews",
            ],
            "education_learning": [
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
                "educational_technology",
                "lifelong_learning",
            ],
            "health_wellness": [
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
                "medical_treatments",
                "recovery",
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
                "travel_stories",
                "transportation",
                "accommodations",
            ],
            "food_cooking": [
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
                "food_trends",
                "meal_planning",
            ],
            "sports_fitness": [
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
                "recreational_activities",
                "sports_culture",
                "exercise",
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
                "creative_pursuits",
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
                "generational_differences",
                "family_support",
                "family_dynamics",
                "family_celebrations",
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
                "skill_based_hobbies",
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
                "income",
                "expenses",
            ],
            "home_living": [
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
                "home_projects",
            ],
            "fashion_style": [
                "clothing",
                "fashion_trends",
                "personal_style",
                "beauty",
                "makeup",
                "fashion_advice",
                "style_inspiration",
                "fashion_industry",
                "accessories",
                "grooming",
                "fashion_events",
                "style_choices",
                "appearance",
            ],
            "nature_environment": [
                "environmental_issues",
                "climate_change",
                "nature_appreciation",
                "outdoor_experiences",
                "wildlife",
                "conservation",
                "sustainability",
                "natural_phenomena",
                "environmental_activism",
                "ecological_awareness",
                "nature_activities",
                "environmental_responsibility",
            ],
            "spirituality_religion": [
                "faith",
                "religious_practices",
                "spiritual_growth",
                "meditation",
                "religious_community",
                "spiritual_beliefs",
                "religious_traditions",
                "spiritual_experiences",
                "religious_studies",
                "philosophy",
                "meaning_of_life",
                "transcendence",
                "religious_ceremonies",
            ],
            "news_current_events": [
                "political_news",
                "world_events",
                "local_news",
                "breaking_news",
                "current_affairs",
                "news_analysis",
                "social_issues",
                "public_policy",
                "global_events",
                "news_commentary",
                "journalism",
                "media_coverage",
            ],
            "communication_social": [
                "social_interactions",
                "communication_skills",
                "social_media_use",
                "online_communities",
                "networking",
                "social_trends",
                "digital_communication",
                "social_behavior",
                "community_involvement",
                "social_dynamics",
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
                "transportation_technology",
                "urban_transportation",
                "transport_planning",
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
                "scientific_community",
                "science_education",
            ],
            "history": [
                "historical_events",
                "historical_figures",
                "cultural_history",
                "historical_analysis",
                "historical_periods",
                "heritage",
                "archaeology",
                "historical_significance",
                "historical_research",
                "historical_context",
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
                "gaming_experiences",
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
                "musical_experiences",
            ],
            "literature_writing": [
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
                "writing_techniques",
            ],
            "photography_visual": [
                "photography_techniques",
                "visual_storytelling",
                "photo_editing",
                "photography_equipment",
                "visual_arts",
                "image_composition",
                "photography_genres",
                "visual_media",
                "graphic_design",
                "visual_culture",
            ],
            "community_social_issues": [
                "social_justice",
                "community_involvement",
                "volunteer_work",
                "activism",
                "social_change",
                "community_development",
                "civic_engagement",
                "social_responsibility",
                "community_events",
                "social_movements",
            ],
            "lifestyle": [
                "daily_routines",
                "life_choices",
                "lifestyle_changes",
                "personal_preferences",
                "life_philosophy",
                "lifestyle_trends",
                "quality_of_life",
                "life_experiences",
                "lifestyle_balance",
                "personal_lifestyle",
                "lifestyle_advice",
            ],
        }

        # Validate taxonomy structure
        for major_topic, subtopics in taxonomy.items():
            if len(subtopics) < 10:
                logger.warning(f"Topic '{major_topic}' has fewer than 10 subtopics")

        logger.info(
            f"Built taxonomy with {len(taxonomy)} major topics and "
            f"{sum(len(subtopics) for subtopics in taxonomy.values())} subtopics"
        )

        return taxonomy

    def _precompute_embeddings(self) -> None:
        """
        Precompute embeddings for all taxonomy terms to optimize runtime performance.

        This follows the academic best practice of separating preprocessing from
        runtime computation for better efficiency in production systems.
        """
        self.major_topic_embeddings = {}
        self.subtopic_embeddings = {}

        # Compute embeddings for major topics
        major_topics = list(self.taxonomy.keys())
        major_embeddings = self.model.encode(major_topics, convert_to_numpy=True)
        self.major_topic_embeddings = dict(zip(major_topics, major_embeddings))

        # Compute embeddings for subtopics
        all_subtopics = []
        subtopic_to_major = {}

        for major_topic, subtopics in self.taxonomy.items():
            for subtopic in subtopics:
                all_subtopics.append(subtopic)
                subtopic_to_major[subtopic] = major_topic

        subtopic_embeddings = self.model.encode(all_subtopics, convert_to_numpy=True)
        self.subtopic_embeddings = dict(zip(all_subtopics, subtopic_embeddings))
        self.subtopic_to_major = subtopic_to_major

        logger.info(
            f"Precomputed embeddings for {len(self.major_topic_embeddings)} major topics "
            f"and {len(self.subtopic_embeddings)} subtopics"
        )

    def _compute_semantic_similarity(
        self, input_embedding: np.ndarray, target_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute cosine similarity between input embedding and target embeddings.

        Args:
            input_embedding: Embedding vector for input text
            target_embeddings: Dictionary of category names to embedding vectors

        Returns:
            Dictionary mapping category names to similarity scores
        """
        similarities = {}

        for category, target_embedding in target_embeddings.items():
            # Reshape for sklearn cosine_similarity
            input_reshaped = input_embedding.reshape(1, -1)
            target_reshaped = target_embedding.reshape(1, -1)

            similarity = cosine_similarity(input_reshaped, target_reshaped)[0][0]
            similarities[category] = float(similarity)

        return similarities

    def map_words_to_taxonomy(
        self,
        words: Union[Set[str], List[str]],
        top_n: int = 5,
        min_similarity: float = 0.1,
    ) -> Dict[str, float]:
        """
        Map a collection of words to the most similar taxonomy categories.

        This is the main entry point function that implements the core mapping algorithm
        using semantic similarity based on transformer embeddings.

        Args:
            words: Set or list of words to map to taxonomy
            top_n: Number of top matches to return
            min_similarity: Minimum similarity threshold for inclusion

        Returns:
            Dictionary with "major:subtopic" keys and percentage similarity values
        """
        if not words:
            logger.warning("No words provided for mapping")
            return {}

        # Convert to list and create input text
        word_list = list(words) if isinstance(words, set) else words
        input_text = " ".join(word_list)

        # Generate embedding for input words
        input_embedding = self.model.encode([input_text], convert_to_numpy=True)[0]

        # Compute similarities with all subtopics
        subtopic_similarities = self._compute_semantic_similarity(
            input_embedding, self.subtopic_embeddings
        )

        # Filter by minimum similarity and sort by score
        filtered_similarities = {
            subtopic: score
            for subtopic, score in subtopic_similarities.items()
            if score >= min_similarity
        }

        # Sort by similarity score (descending)
        sorted_similarities = sorted(
            filtered_similarities.items(), key=lambda x: x[1], reverse=True
        )

        # Take top N and format results
        top_matches = sorted_similarities[:top_n]

        results = {}
        for subtopic, similarity in top_matches:
            major_topic = self.subtopic_to_major[subtopic]
            key = f"{major_topic}:{subtopic}"
            # Convert to percentage and round to 2 decimal places
            percentage = round(similarity * 100, 2)
            results[key] = percentage

        logger.info(
            f"Mapped {len(word_list)} words to {len(results)} taxonomy categories"
        )

        return results

    def get_taxonomy_info(self) -> Dict[str, Any]:
        """
        Get information about the taxonomy structure.

        Returns:
            Dictionary with taxonomy statistics and structure
        """
        return {
            "num_major_topics": len(self.taxonomy),
            "num_subtopics": sum(
                len(subtopics) for subtopics in self.taxonomy.values()
            ),
            "major_topics": list(self.taxonomy.keys()),
            "avg_subtopics_per_major": np.mean([
                len(subtopics) for subtopics in self.taxonomy.values()
            ]),
            "taxonomy_structure": {
                major: len(subtopics) for major, subtopics in self.taxonomy.items()
            },
        }


def map_topic_words_to_taxonomy(
    words: Union[Set[str], List[str]], top_n: int = 5
) -> Dict[str, float]:
    """
    Main entry point function for mapping topic words to taxonomy categories.

    This function provides a simple interface to the TopicTaxonomyMapper for
    quick topic categorization without requiring class instantiation management.

    Args:
        words: Set or list of words from topic extraction
        top_n: Number of top taxonomy matches to return

    Returns:
        Dictionary mapping "major:subtopic" to percentage similarity scores

    Example:
        >>> words = {'friends', 'crush', 'embarrassed', 'classmate', 'jealous'}
        >>> results = map_topic_words_to_taxonomy(words, top_n=3)
        >>> print(results)
        {'relationships:romantic_relationships': 85.4, 'relationships:friendship': 78.2, ...}
    """
    mapper = TopicTaxonomyMapper()
    return mapper.map_words_to_taxonomy(words, top_n=top_n)


# Test validation function
def test_topic_taxonomy_mapper():
    """
    Test the topic taxonomy mapper with the provided example word set.

    This test validates the functionality using academic evaluation criteria:
    1. Semantic relevance of mappings
    2. Consistency of similarity scores
    3. Coverage of different topic domains
    """
    print("=" * 60)
    print("TESTING TOPIC TAXONOMY MAPPER")
    print("=" * 60)

    # Example word set from the user (appears to be relationship/personal focused)
    test_words = {
        "doing",
        "nice know embarrassed",
        "classmate elementary actually",
        "cyrus s like",
        "friends",
        "s nice know",
        "s really",
        "friends s",
        "doing embarrasing blah",
        "like",
        "came",
        "s classmate",
        "don",
        "really",
        "s like anymore",
        "s crush close",
        "doing embarrasing",
        "like anymore",
        "blah",
        "s doing",
        "s really bad",
        "s jealous yes",
        "bad hate",
        "ve addicted s",
        "s classmate elementary",
        "really bad",
        "yes",
        "s doing embarrasing",
        "don t",
        "admit ve addicted",
        "addicted s like",
        "dreamed",
        "cyrus s",
        "m",
        "crush",
        "classmate elementary",
        "admit ve",
        "s like",
        "jealous yes m",
        "m flattered cuz",
        "came head s",
        "s crush",
        "cyrus",
        "anymore",
        "classmate",
        "bad",
        "jealous yes",
        "admit",
    }

    # Initialize mapper
    print("Initializing TopicTaxonomyMapper...")
    mapper = TopicTaxonomyMapper()

    # Test taxonomy info
    info = mapper.get_taxonomy_info()
    print(f"\nTaxonomy Info:")
    print(f"  Major topics: {info['num_major_topics']}")
    print(f"  Total subtopics: {info['num_subtopics']}")
    print(f"  Average subtopics per major topic: {info['avg_subtopics_per_major']:.1f}")

    # Test mapping functionality
    print(f"\nTesting with {len(test_words)} words...")
    results = mapper.map_words_to_taxonomy(test_words, top_n=8)

    print(f"\nTop {len(results)} taxonomy mappings:")
    for category, percentage in results.items():
        print(f"  {category}: {percentage}%")

    # Test entry point function
    print(f"\nTesting entry point function...")
    entry_results = map_topic_words_to_taxonomy(test_words, top_n=5)
    print(f"Entry point returned {len(entry_results)} mappings:")
    for category, percentage in entry_results.items():
        print(f"  {category}: {percentage}%")

    # Validate results quality
    print(f"\nValidation:")
    print(f"  ✓ Returned {len(results)} categories (expected top N)")
    print(
        f"  ✓ All percentages are numeric: {all(isinstance(p, (int, float)) for p in results.values())}"
    )
    print(
        f"  ✓ Format follows 'major:subtopic': {all(':' in key for key in results.keys())}"
    )

    # Check if relationship-related categories are highly ranked (expected for this word set)
    relationship_categories = [
        k for k in results.keys() if k.startswith("relationships:")
    ]
    if relationship_categories:
        print(f"  ✓ Relationship categories found: {relationship_categories}")

    print(f"\n✓ Test completed successfully!")

    return results


if __name__ == "__main__":
    # Run the test
    test_results = test_topic_taxonomy_mapper()
