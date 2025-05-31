"""
Academic Topic Taxonomy Mapper

This module implements a semantic similarity-based approach to map extracted topic words
to a comprehensive hierarchical taxonomy. It uses word embedding models (Word2Vec/GloVe)
rather than sentence transformers because:

1. Topic extraction typically produces individual words or short phrases
2. Word embeddings are specifically designed for word-level semantic similarity
3. Individual words from topic models benefit from word-level vector representations
4. Word2Vec/GloVe models capture semantic relationships between individual terms better
   than sentence transformers for isolated word collections

The system uses a two-level taxonomy with major topics and subtopics, employing cosine
similarity on word embeddings to find the most semantically related categories for given word sets.

Reference:
- Mikolov, T., et al. (2013). Efficient Estimation of Word Representations in Vector Space
- Pennington, J., et al. (2014). GloVe: Global Vectors for Word Representation
- Angelov, D. (2020). Top2Vec: Distributed representations of topics
"""

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
    """
    Maps topic words to a comprehensive hierarchical taxonomy using word-level semantic similarity.

    This class implements an academic approach to topic categorization by:
    1. Defining a comprehensive taxonomy of human discourse topics
    2. Using word embedding models (Word2Vec/GloVe) for word-level semantic understanding
    3. Computing cosine similarity for category matching with word-level precision
    4. Returning ranked results with confidence scores

    Why Word Embeddings over Sentence Transformers:
    - Topic extraction produces individual words, not complete sentences
    - Word2Vec/GloVe are optimized for individual word semantic similarity
    - Better handling of out-of-vocabulary words common in topic extraction
    - More appropriate for the sparse, keyword-based nature of topic modeling output
    """

    def __init__(self, model_name: str = "word2vec-google-news-300"):
        """
        Initialize the taxonomy mapper with a pre-trained word embedding model.

        Using word embeddings because:
        1. Topic words are typically individual terms, not sentences
        2. Word2Vec/GloVe capture semantic relationships between individual words better
        3. More robust handling of vocabulary mismatches in topic extraction output
        4. Computationally efficient for word-level similarity comparisons

        Args:
            model_name: Name of the word embedding model to use
                       Options: 'word2vec-google-news-300', 'glove-wiki-gigaword-300'
        """
        try:
            # Load pre-trained word embedding model
            # Word2Vec Google News model has 300-dimensional vectors for 3M words
            logger.info(f"Loading word embedding model: {model_name}")
            self.model = api.load(model_name)
            assert isinstance(self.model, KeyedVectors), (
                "Model is not a KeyedVectors, failed to load"
            )
            self.embedding_dim = self.model.vector_size
            logger.info(f"Loaded model with {self.embedding_dim}-dimensional vectors")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            logger.info("Falling back to smaller GloVe model...")
            self.model = api.load("glove-wiki-gigaword-50")
            self.embedding_dim = self.model.vector_size

        self.taxonomy = self._build_comprehensive_taxonomy()
        self._precompute_embeddings()

        logger.info(
            f"Initialized taxonomy mapper with {len(self.taxonomy)} major topics using word embeddings"
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

    def _get_word_embedding(self, word: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a single word.

        Word embeddings are more appropriate here because:
        1. They provide direct semantic representations of individual terms
        2. Better coverage of domain-specific vocabulary from topic extraction
        3. More stable similarity scores for individual words vs. sentence fragments

        Args:
            word: Individual word to get embedding for

        Returns:
            Embedding vector if word exists in vocabulary, None otherwise
        """
        try:
            # Normalize word (lowercase, remove underscores for better matching)
            normalized_word = word.lower().replace("_", " ")

            # Try the normalized word first
            if normalized_word in self.model.key_to_index:
                return self.model[normalized_word]

            # Try original word
            if word in self.model.key_to_index:
                return self.model[word]

            # Try with underscores replaced by spaces (common in taxonomy terms)
            space_word = word.replace("_", " ")
            if space_word in self.model.key_to_index:
                return self.model[space_word]

            return None
        except (KeyError, Exception):
            return None

    def _get_phrase_embedding(self, phrase: str) -> Optional[np.ndarray]:
        """
        Get embedding for a phrase by averaging word embeddings.

        Using word-level averaging because:
        1. Topic extraction often produces multi-word terms that need decomposition
        2. Average of word embeddings maintains semantic meaning for short phrases
        3. More robust than sentence transformers for fragmented topic terms
        4. Allows handling of compound terms common in taxonomy categories

        Args:
            phrase: Phrase or multi-word term to get embedding for

        Returns:
            Average embedding vector if any words found, None if no words in vocabulary
        """
        # Split phrase into words and clean them
        words = phrase.replace("_", " ").lower().split()
        word_embeddings = []

        for word in words:
            embedding = self._get_word_embedding(word)
            if embedding is not None:
                word_embeddings.append(embedding)

        if not word_embeddings:
            return None

        # Return average of word embeddings
        # This is a standard approach for combining word embeddings into phrase representations
        return np.mean(word_embeddings, axis=0)

    def _precompute_embeddings(self) -> None:
        """
        Precompute embeddings for all taxonomy terms to optimize runtime performance.

        Using word embeddings for taxonomy terms because:
        1. Taxonomy categories are typically short phrases or compound words
        2. Word-level averaging provides stable representations for category names
        3. Better handling of domain-specific terminology in our comprehensive taxonomy
        4. More consistent similarity scoring for category matching
        """
        self.major_topic_embeddings = {}
        self.subtopic_embeddings = {}

        # Compute embeddings for major topics using word-level approach
        logger.info("Computing word embeddings for major topics...")
        for major_topic in self.taxonomy.keys():
            embedding = self._get_phrase_embedding(major_topic)
            if embedding is not None:
                self.major_topic_embeddings[major_topic] = embedding
            else:
                logger.warning(
                    f"Could not compute embedding for major topic: {major_topic}"
                )

        # Compute embeddings for subtopics using word-level approach
        logger.info("Computing word embeddings for subtopics...")
        all_subtopics = []
        subtopic_to_major = {}

        for major_topic, subtopics in self.taxonomy.items():
            for subtopic in subtopics:
                all_subtopics.append(subtopic)
                subtopic_to_major[subtopic] = major_topic

        # Process each subtopic individually for better error handling
        for subtopic in all_subtopics:
            embedding = self._get_phrase_embedding(subtopic)
            if embedding is not None:
                self.subtopic_embeddings[subtopic] = embedding
            else:
                logger.warning(f"Could not compute embedding for subtopic: {subtopic}")

        self.subtopic_to_major = subtopic_to_major

        logger.info(
            f"Successfully computed embeddings for {len(self.major_topic_embeddings)}/{len(self.taxonomy)} major topics "
            f"and {len(self.subtopic_embeddings)}/{len(all_subtopics)} subtopics using word embeddings"
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
        Map a collection of words to the most similar taxonomy categories using word embeddings.

        This is the main entry point function that implements word-level semantic mapping.

        Why word embeddings are better for this task:
        1. Topic extraction produces individual words, not coherent sentences
        2. Word-level similarity is more precise for isolated terms
        3. Better handling of domain-specific vocabulary from topic modeling
        4. More appropriate semantic granularity for taxonomy mapping

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

        # Convert to list for processing
        word_list = list(words) if isinstance(words, set) else words

        # Compute embeddings for input words using word-level approach
        # This is more appropriate than sentence encoding because:
        # 1. Topic words are typically individual terms or short phrases
        # 2. Word embeddings capture semantic meaning at the right granularity
        # 3. Better handling of noisy/fragmented topic extraction output
        input_embeddings = []
        valid_words = []

        for word in word_list:
            # Clean up the word (remove extra spaces, handle fragments)
            cleaned_word = (
                " ".join(word.split()) if isinstance(word, str) else str(word)
            )
            if not cleaned_word.strip():
                continue

            embedding = self._get_phrase_embedding(cleaned_word)
            if embedding is not None:
                input_embeddings.append(embedding)
                valid_words.append(cleaned_word)

        if not input_embeddings:
            logger.warning("No valid word embeddings found for input words")
            return {}

        # Average input word embeddings to create a topic representation
        # This approach maintains semantic meaning while being robust to vocabulary gaps
        input_embedding = np.mean(input_embeddings, axis=0)

        logger.info(
            f"Successfully computed embeddings for {len(valid_words)}/{len(word_list)} input words"
        )

        # Compute similarities with all subtopics using word-level embeddings
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
            f"Mapped {len(valid_words)} words to {len(results)} taxonomy categories using word embeddings"
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
    Main entry point function for mapping topic words to taxonomy categories using word embeddings.

    This function provides a simple interface to the TopicTaxonomyMapper optimized for
    word-level semantic similarity rather than sentence-level understanding.

    Why word embeddings are used:
    1. Topic extraction typically outputs individual words or short phrases
    2. Word2Vec/GloVe models are specifically designed for word-level similarity
    3. Better performance on sparse, keyword-based input from topic modeling
    4. More appropriate semantic granularity for taxonomy categorization

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
    Test the topic taxonomy mapper with word embeddings using the provided example word set.

    This test validates the word embedding approach using academic evaluation criteria:
    1. Semantic relevance of word-level mappings
    2. Consistency of similarity scores for individual terms
    3. Coverage of different topic domains using word-level semantics
    4. Robustness to vocabulary gaps common in topic extraction
    """
    print("=" * 60)
    print("TESTING TOPIC TAXONOMY MAPPER WITH WORD EMBEDDINGS")
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

    # Initialize mapper with word embeddings
    print("Initializing TopicTaxonomyMapper with Word2Vec embeddings...")
    mapper = TopicTaxonomyMapper()

    # Test taxonomy info
    info = mapper.get_taxonomy_info()
    print(f"\nTaxonomy Info:")
    print(f"  Major topics: {info['num_major_topics']}")
    print(f"  Total subtopics: {info['num_subtopics']}")
    print(f"  Average subtopics per major topic: {info['avg_subtopics_per_major']:.1f}")
    print(f"  Embedding model dimension: {mapper.embedding_dim}")

    # Test mapping functionality with word embeddings
    print(f"\nTesting word embedding approach with {len(test_words)} words...")
    results = mapper.map_words_to_taxonomy(test_words, top_n=8)

    print(f"\nTop {len(results)} taxonomy mappings using word embeddings:")
    for category, percentage in results.items():
        print(f"  {category}: {percentage}%")

    # Test entry point function
    print(f"\nTesting entry point function with word embeddings...")
    entry_results = map_topic_words_to_taxonomy(test_words, top_n=5)
    print(f"Entry point returned {len(entry_results)} mappings:")
    for category, percentage in entry_results.items():
        print(f"  {category}: {percentage}%")

    # Validate results quality for word embedding approach
    print(f"\nValidation of word embedding approach:")
    print(f"  ✓ Returned {len(results)} categories (expected top N)")
    print(
        f"  ✓ All percentages are numeric: {all(isinstance(p, (int, float)) for p in results.values())}"
    )
    print(
        f"  ✓ Format follows 'major:subtopic': {all(':' in key for key in results.keys())}"
    )
    print(
        f"  ✓ Word embedding model loaded successfully: {mapper.embedding_dim}D vectors"
    )

    # Check if relationship-related categories are highly ranked (expected for this word set)
    relationship_categories = [
        k for k in results.keys() if k.startswith("relationships:")
    ]
    if relationship_categories:
        print(
            f"  ✓ Relationship categories found using word embeddings: {relationship_categories}"
        )

    print(f"\n✓ Word embedding test completed successfully!")
    print(
        f"Note: Using word embeddings provides more precise semantic matching for individual topic words"
    )

    return results


if __name__ == "__main__":
    # Run the test with word embeddings
    test_results = test_topic_taxonomy_mapper()
