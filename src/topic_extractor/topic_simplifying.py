"""
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

    model: KeyedVectors

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
            try:
                self.model = api.load("glove-wiki-gigaword-50")
                assert isinstance(self.model, KeyedVectors), (
                    "Fallback model is not a KeyedVectors, failed to load"
                )
                self.embedding_dim = self.model.vector_size
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback model: {fallback_error}")
                raise RuntimeError(
                    "Could not load any word embedding model"
                ) from fallback_error

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
                "work",
                "marriage",
                "breakups",
                "crushes",
                "friendship",
                "pets",
                "family_relationships",
                "social_connections",
                "interpersonal_conflict",
                "love",
                "intimacy",
                "relationship_advice",
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
                "spiritual_guidance",
            ],
            "politics": [
                "political_commentary",
                "political_news",
                "political_analysis",
                "political_opinion",
                "republican",
                "democrat",
                "conservative",
                "liberal",
                "socialism",
                "capitalism",
                "social_justice",
            ],
            "self_improvement": [
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
                "career",
                "workplace_issues",
                "professional_skills",
                "entrepreneurship",
                "business",
                "management",
                "leadership",
                "workplace_relationships",
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
            "sports": [
                "training",
                "fitness",
                "health",
                "wellness",
                "nutrition",
                "diet",
                "exercise",
                "football",
                "basketball",
                "soccer",
                "tennis",
                "baseball",
                "running",
                "cycling",
                "swimming",
                "golf",
                "tennis",
                "basketball",
                "soccer",
                "rugby",
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
                "drug_use",
                "alcohol_use",
                "smoking",
                "personal_preferences",
                "lifestyle_trends",
                "quality_of_life",
                "life_experiences",
                "lifestyle_balance",
                "personal_lifestyle",
                "lifestyle_advice",
            ],
        }

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
    mapper: TopicTaxonomyMapper, words: Union[Set[str], List[str]], top_n: int = 5
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
        mapper: TopicTaxonomyMapper instance
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
    return mapper.map_words_to_taxonomy(words, top_n=top_n)


def map_lda_results_to_taxonomy(
    mapper: TopicTaxonomyMapper,
    lda_results: Dict[str, Any],
    top_n: int = 10,
    weight_threshold: float = 0.2,
    min_similarity: float = 0.05,
) -> Dict[str, Any]:
    """
    Map LDA analysis results to taxonomy categories using improved weighted word importance
    and topic quality scoring.

    This enhanced implementation:
    1. Uses weighted embeddings that combine LDA weights with coherence scores
    2. Applies topic quality as a confidence multiplier
    3. Uses percentile-based word selection for better focus
    4. Returns more comprehensive results above specified threshold

    Args:
        mapper: TopicTaxonomyMapper instance
        lda_results: Dictionary containing LDA analysis results with topics, words, and weights
        top_n: Number of top taxonomy matches to consider per topic (increased for better coverage)
        weight_threshold: Minimum weight threshold for including words (used as percentile)
        min_similarity: Minimum similarity threshold after quality weighting

    Returns:
        Dictionary mapping "major:subtopic" to percentage similarity scores

    Example:
        >>> lda_data = {
        ...     "lda_results": {
        ...         "high_quality_topics": [...],
        ...         "topics": [...]
        ...     }
        ... }
        >>> results = map_lda_results_to_taxonomy(mapper, lda_data)
    """
    if not lda_results or "lda_results" not in lda_results:
        logger.error("Invalid LDA results structure - missing 'lda_results' key")
        return {"error": "Invalid LDA results structure"}

    lda_data = lda_results["lda_results"]

    # Determine which topics to use: high quality topics if available, otherwise all topics
    topics_to_analyze = []
    if "high_quality_topics" in lda_data and lda_data["high_quality_topics"]:
        topics_to_analyze = lda_data["high_quality_topics"]
        topics_source = "high_quality_topics"
        logger.info(f"Using {len(topics_to_analyze)} high quality topics for analysis")
    elif "topics" in lda_data and lda_data["topics"]:
        topics_to_analyze = lda_data["topics"]
        topics_source = "all_topics"
        logger.info(
            f"No high quality topics found, using all {len(topics_to_analyze)} topics"
        )
    else:
        logger.error("No topics found in LDA results")
        return {"error": "No topics found in LDA results"}

    # Process each topic with improved weighting
    all_results = {}

    for topic in topics_to_analyze:
        topic_id = topic.get("topic_id", "unknown")
        topic_label = topic.get("label", "unlabeled")
        words = topic.get("words", [])
        weights = topic.get("weights", [])
        coherence_scores = topic.get("coherence_scores", [])
        topic_quality = topic.get("topic_quality", 0.5)
        semantic_coherence = topic.get("semantic_coherence", 0.5)

        if not words or not weights or len(words) != len(weights):
            logger.warning(f"Topic {topic_id} has mismatched words/weights, skipping")
            continue

        # Use coherence scores if available, otherwise default to moderate values
        if not coherence_scores or len(coherence_scores) != len(words):
            coherence_scores = [0.5] * len(words)
            logger.info(f"Topic {topic_id}: Using default coherence scores")

        # Improved word selection using percentile-based threshold
        # Convert weight_threshold to percentile (0.2 -> 60th percentile)
        weight_percentile = max(50, (1 - weight_threshold) * 100)
        weight_threshold_value = np.percentile(weights, weight_percentile)

        # Ensure minimum number of words (at least top 3)
        min_words = min(3, len(words))

        # Get word-weight-coherence tuples and sort by weight
        word_data = list(zip(words, weights, coherence_scores))
        word_data.sort(key=lambda x: x[1], reverse=True)

        # Select words either above threshold or top N
        selected_data = []
        for word, weight, coherence in word_data:
            if weight >= weight_threshold_value or len(selected_data) < min_words:
                selected_data.append((word, weight, coherence))

        # Limit selection to avoid noise (max 8 words per topic)
        selected_data = selected_data[:8]

        selected_words = [item[0] for item in selected_data]
        selected_weights = [item[1] for item in selected_data]
        selected_coherence = [item[2] for item in selected_data]

        if not selected_words:
            logger.warning(f"Topic {topic_id} has no valid words after filtering")
            continue

        logger.info(
            f"Topic {topic_id}: Analyzing {len(selected_words)} weighted words from {len(words)} total"
        )

        # Create improved weighted embedding
        topic_embedding = _create_weighted_topic_embedding(
            selected_words, selected_weights, selected_coherence, mapper
        )

        if topic_embedding is None:
            logger.warning(f"Topic {topic_id}: No valid embeddings found")
            continue

        # Compute similarities with subtopics
        subtopic_similarities = mapper._compute_semantic_similarity(
            topic_embedding, mapper.subtopic_embeddings
        )

        # Apply topic quality weighting to similarities
        # Use a scaled quality multiplier to maintain reasonable score ranges
        quality_multiplier = 0.5 + (topic_quality * 0.5)  # Scale to 0.5-1.0 range

        weighted_similarities = {
            subtopic: similarity * quality_multiplier
            for subtopic, similarity in subtopic_similarities.items()
        }

        # Filter by minimum similarity threshold
        filtered_similarities = {
            subtopic: score
            for subtopic, score in weighted_similarities.items()
            if score >= min_similarity
        }

        # Convert to major:subtopic format and add to results
        for subtopic, similarity in filtered_similarities.items():
            major_topic = mapper.subtopic_to_major[subtopic]
            key = f"{major_topic}:{subtopic}"
            percentage = round(similarity * 100, 2)

            # Keep the highest score if duplicate keys exist across topics
            if key not in all_results or percentage > all_results[key]:
                all_results[key] = percentage

    # Apply final filtering: keep results >= 20% (more permissive than 25%)
    # and return top results ordered by score
    filtered_results = {k: v for k, v in all_results.items() if v >= 20.0}

    # Limit to top N*2 results to prevent overwhelming output while being comprehensive
    sorted_results = sorted(filtered_results.items(), key=lambda x: x[1], reverse=True)
    final_results = dict(sorted_results[: top_n * 2])

    logger.info(
        f"Successfully mapped LDA topics to {len(final_results)} taxonomy categories "
        f"(from {len(all_results)} total matches)"
    )

    return final_results


def _create_weighted_topic_embedding(
    words: List[str],
    weights: List[float],
    coherence_scores: List[float],
    mapper: TopicTaxonomyMapper,
) -> Optional[np.ndarray]:
    """
    Create weighted topic embedding that combines LDA weights with coherence scores.

    This approach provides more accurate topic representations by:
    1. Weighting words by both their LDA importance and semantic coherence
    2. Using normalized weighted averages instead of simple averages
    3. Handling vocabulary gaps gracefully

    Args:
        words: List of topic words
        weights: LDA weights for each word
        coherence_scores: Coherence scores for each word
        mapper: TopicTaxonomyMapper instance

    Returns:
        Weighted average embedding or None if no valid embeddings found
    """
    embeddings = []
    final_weights = []

    for word, weight, coherence in zip(words, weights, coherence_scores):
        embedding = mapper._get_phrase_embedding(word)
        if embedding is not None:
            # Combine LDA weight with coherence score
            # Higher weight = more important in topic
            # Higher coherence = more semantically consistent
            combined_weight = weight * (
                coherence + 0.2
            )  # Add 0.2 to avoid zero multiplication
            embeddings.append(embedding)
            final_weights.append(combined_weight)

    if not embeddings:
        return None

    # Normalize weights to sum to 1
    final_weights = np.array(final_weights)
    final_weights = final_weights / np.sum(final_weights)

    # Compute weighted average embedding
    weighted_embedding = np.average(embeddings, axis=0, weights=final_weights)

    return weighted_embedding
