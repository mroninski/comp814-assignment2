"""
Semantic similarity-based topic mapping to hierarchical taxonomy using word embeddings.
"""

import csv
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, cast
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

import numpy as np
from gensim.models import KeyedVectors
from gensim import downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IABTaxonomyFetcher:
    """Fetches and caches IAB Content Taxonomy from GitHub."""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "iab_taxonomy.json")
        self.cache_expiry_days = 7
        self.iab_tsv_url = "https://raw.githubusercontent.com/InteractiveAdvertisingBureau/Taxonomies/develop/Content%20Taxonomies/Content%20Taxonomy%203.1.tsv"

        os.makedirs(cache_dir, exist_ok=True)

    def _is_cache_valid(self) -> bool:
        """Check if cached taxonomy is still valid."""
        if not os.path.exists(self.cache_file):
            return False

        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            cache_timestamp = datetime.fromisoformat(cache_data.get("timestamp", ""))
            expiry_time = cache_timestamp + timedelta(days=self.cache_expiry_days)

            return datetime.now() < expiry_time
        except (json.JSONDecodeError, ValueError, KeyError):
            return False

    def _load_from_cache(self) -> Dict[str, List[str]]:
        """Load taxonomy from cache."""
        with open(self.cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        return cache_data["taxonomy"]

    def _save_to_cache(self, taxonomy: Dict[str, List[str]]) -> None:
        """Save taxonomy to cache."""
        cache_data = {"timestamp": datetime.now().isoformat(), "taxonomy": taxonomy}
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)

    def _fetch_tsv_data(self) -> str:
        """Fetch TSV data from IAB GitHub repository."""
        try:
            logger.info(f"Fetching IAB taxonomy from {self.iab_tsv_url}")
            with urlopen(self.iab_tsv_url, timeout=30) as response:
                return response.read().decode("utf-8")
        except (URLError, HTTPError) as e:
            logger.error(f"Failed to fetch IAB taxonomy: {e}")
            raise RuntimeError(f"Could not fetch IAB taxonomy: {e}")

    def _parse_tsv_to_taxonomy(self, tsv_content: str) -> Dict[str, List[str]]:
        """Parse TSV content into taxonomy structure."""
        taxonomy = {}
        lines = tsv_content.strip().split("\n")

        if len(lines) < 3:
            raise ValueError("Invalid TSV format: insufficient lines")

        # Skip header lines (first 2 lines)
        csv_reader = csv.reader(lines[2:], delimiter="\t")

        for row in csv_reader:
            if len(row) < 4:
                continue

            # Extract Tier 1 (category) and Name (subtopic)
            tier1 = row[3].strip() if len(row) > 3 else ""
            name = row[2].strip() if len(row) > 2 else ""

            if not tier1 or not name:
                continue

            # Normalize category name for consistency
            category = tier1.lower().replace(" ", "_").replace("&", "and")

            # Skip if name is identical to tier1 (these are category headers)
            if name.lower() == tier1.lower():
                continue

            # Initialize category if not exists
            if category not in taxonomy:
                taxonomy[category] = []

            # Add subtopic, avoiding duplicates
            subtopic = name.lower().replace(" ", "_").replace("&", "and")
            if subtopic not in taxonomy[category]:
                taxonomy[category].append(subtopic)

        # Filter out categories with no subtopics
        taxonomy = {k: v for k, v in taxonomy.items() if v}

        logger.info(
            f"Parsed IAB taxonomy with {len(taxonomy)} categories and "
            f"{sum(len(subtopics) for subtopics in taxonomy.values())} subtopics"
        )

        return taxonomy

    def get_taxonomy(self) -> Dict[str, List[str]]:
        """Get IAB taxonomy, using cache if valid or fetching fresh data."""
        if self._is_cache_valid():
            logger.info("Loading IAB taxonomy from cache")
            return self._load_from_cache()

        logger.info("Fetching fresh IAB taxonomy")
        tsv_content = self._fetch_tsv_data()
        taxonomy = self._parse_tsv_to_taxonomy(tsv_content)
        self._save_to_cache(taxonomy)

        return taxonomy


class TopicTaxonomyMapper:
    """Maps topic words to hierarchical taxonomy using word embeddings.
    Details on the model:
    https://github.com/piskvorky/gensim-data?tab=readme-ov-file

    """

    def __init__(self, model_name: str = "glove-twitter-200"):
        """Initialize with pre-trained word embedding model."""
        try:
            logger.info(f"Loading word embedding model: {model_name}")
            self.model: KeyedVectors = cast(KeyedVectors, api.load(model_name))
            self.embedding_dim: int = self.model.vector_size
            logger.info(f"Loaded model with {self.embedding_dim}-dimensional vectors")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            logger.info("Falling back to smaller GloVe model...")
            try:
                self.model: KeyedVectors = cast(
                    KeyedVectors, api.load("glove-wiki-gigaword-50")
                )
                self.embedding_dim: int = self.model.vector_size
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback model: {fallback_error}")
                raise RuntimeError(
                    "Could not load any word embedding model"
                ) from fallback_error

        self.iab_fetcher = IABTaxonomyFetcher()
        self.taxonomy: Dict[str, List[str]] = self._build_taxonomy()
        self._precompute_embeddings()
        logger.info(
            f"Initialized taxonomy mapper with {len(self.taxonomy)} major topics"
        )

    def _build_taxonomy(self) -> Dict[str, List[str]]:
        """Build taxonomy from IAB Content Categories."""
        try:
            taxonomy = self.iab_fetcher.get_taxonomy()
            logger.info(
                f"Built IAB taxonomy with {len(taxonomy)} major topics and "
                f"{sum(len(subtopics) for subtopics in taxonomy.values())} subtopics"
            )
            return taxonomy
        except Exception as e:
            logger.error(f"Failed to load IAB taxonomy: {e}")
            logger.warning("Falling back to basic taxonomy")

            # Minimal fallback taxonomy in case IAB fetch fails
            fallback_taxonomy = {
                "automotive": ["cars", "trucks", "vehicles", "driving"],
                "business": ["finance", "work", "career", "money"],
                "entertainment": ["movies", "music", "tv", "games"],
                "food": ["cooking", "recipes", "restaurants", "dining"],
                "health": ["fitness", "nutrition", "medical", "wellness"],
                "sports": ["football", "basketball", "soccer", "athletics"],
                "technology": ["computers", "internet", "software", "tech"],
                "travel": ["vacation", "tourism", "destinations", "trips"],
            }

            logger.info(
                f"Using fallback taxonomy with {len(fallback_taxonomy)} categories"
            )
            return fallback_taxonomy

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
                # Prepare subtopic for embedding
                subtopic = subtopic.replace("/", " ").replace("_", " ")
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
    top_n: int = 200,
    min_similarity: float = 0.30,
) -> Dict[str, float]:
    """
    Map LDA analysis results to taxonomy categories using weighted word embeddings.

    Args:
        mapper: TopicTaxonomyMapper instance
        lda_results: Dictionary containing LDA analysis results
        top_n: Number of top taxonomy matches to return

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

        # Select top words by weight (at least top 3)
        word_data = list(zip(words, weights, coherence_scores))
        word_data.sort(key=lambda x: x[1], reverse=True)

        # Take top 60th percentile or minimum 3 words
        threshold = np.percentile(weights, 60) if len(weights) > 3 else min(weights)
        selected_data = [item for item in word_data if item[1] >= threshold]

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

            if weighted_similarity < min_similarity:
                continue

            major_topic = mapper.subtopic_to_major[subtopic]
            key = f"{major_topic}:{subtopic}"
            percentage = round(weighted_similarity * 100, 2)

            # Keep highest score if duplicate keys exist
            if key not in all_results or percentage > all_results[key]:
                all_results[key] = percentage

    # Return top results sorted by score
    sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
    final_results = dict(sorted_results[:top_n])

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
