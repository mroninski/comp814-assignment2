import gensim.downloader as api
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors

from topic_extractor.lda_extraction import extract_topics_lda


class TopicMapper:
    """
    Maps extracted topics to predefined reference topics using embeddings.
    """

    def __init__(self, embedding_model="sentence-bert"):
        """
        Initialize with chosen embedding model.

        Parameters:
        -----------
        embedding_model : str
            Options: 'sentence-bert', 'word2vec', 'glove'
        """
        self.embedding_model = embedding_model
        self.reference_topics = None
        self.reference_embeddings = None

        # Load embedding model
        if embedding_model == "sentence-bert":
            # Best for topic phrases
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        elif embedding_model == "word2vec":
            # Good for single words
            self.model = api.load("word2vec-google-news-300")
        elif embedding_model == "glove":
            # Alternative word embeddings
            self.model = api.load("glove-wiki-gigaword-100")

    def setup_reference_topics(self):
        """
        One-time setup of 100 reference topics.
        These cover common blog topics across demographics.
        """
        # Predefined topic categories (expandable to 100)
        self.reference_topics = [
            # Personal & Relationships
            "family relationships",
            "romantic relationships",
            "friendship",
            "parenting",
            "marriage",
            "dating",
            "breakup",
            "divorce",
            # Education & Career
            "school",
            "university",
            "homework",
            "exams",
            "graduation",
            "job search",
            "career",
            "work stress",
            "colleagues",
            "promotion",
            # Emotions & Mental Health
            "happiness",
            "sadness",
            "anxiety",
            "depression",
            "stress",
            "anger",
            "fear",
            "loneliness",
            "self-esteem",
            "therapy",
            # Daily Life
            "daily routine",
            "weekend activities",
            "hobbies",
            "cooking",
            "shopping",
            "cleaning",
            "commuting",
            "sleep",
            "exercise",
            # Entertainment & Media
            "movies",
            "music",
            "books",
            "television",
            "video games",
            "concerts",
            "sports",
            "celebrities",
            "social media",
            # Technology
            "internet",
            "computers",
            "smartphones",
            "software",
            "gaming",
            "websites",
            "programming",
            "tech news",
            "gadgets",
            # Health & Wellness
            "diet",
            "fitness",
            "illness",
            "medical",
            "nutrition",
            "yoga",
            "meditation",
            "weight loss",
            "healthy lifestyle",
            # Travel & Places
            "vacation",
            "travel",
            "hometown",
            "moving",
            "tourism",
            "restaurants",
            "nightlife",
            "nature",
            "cities",
            # Money & Finance
            "budget",
            "savings",
            "debt",
            "shopping habits",
            "income",
            "expenses",
            "financial stress",
            "investments",
            "student loans",
            # Social Issues & Events
            "politics",
            "current events",
            "social justice",
            "environment",
            "community",
            "volunteering",
            "religion",
            "culture",
            "traditions",
            # Personal Growth
            "goals",
            "dreams",
            "motivation",
            "learning",
            "self-improvement",
            "confidence",
            "decision making",
            "time management",
            "productivity",
            # Specific Demographics
            "teenage life",
            "college life",
            "young adult",
            "midlife",
            "parenthood",
            "retirement",
            "gender issues",
            "generation gap",
        ]

        # Calculate embeddings for reference topics
        self._calculate_reference_embeddings()

    def _calculate_reference_embeddings(self):
        """Calculate embeddings for all reference topics."""
        assert self.reference_topics is not None, (
            "self.reference_topics must be initialized before calculating embeddings. Call setup_reference_topics()."
        )

        if self.embedding_model == "sentence-bert":
            assert isinstance(self.model, SentenceTransformer)
            self.reference_embeddings = self.model.encode(self.reference_topics)
        else:
            assert isinstance(self.model, KeyedVectors)
            embeddings = []
            for topic in self.reference_topics:
                words = topic.split()
                word_vecs = []
                for word in words:
                    if word in self.model:
                        word_vecs.append(self.model[word])
                if word_vecs:
                    embeddings.append(np.mean(word_vecs, axis=0))
                else:
                    embeddings.append(np.zeros(self.model.vector_size))
            self.reference_embeddings = np.array(embeddings)

    def map_to_reference_topic(self, extracted_topic: str, top_k=1):
        """
        Map an extracted topic to the closest reference topic(s).

        Parameters:
        -----------
        extracted_topic : str
            Topic extracted from LDA (e.g., "school homework exam")
        top_k : int
            Number of closest matches to return

        Returns:
        --------
        list : List of tuples (reference_topic, similarity_score)
        """
        topic_embedding: np.ndarray

        if self.embedding_model == "sentence-bert":
            assert isinstance(self.model, SentenceTransformer)
            topic_embedding = self.model.encode([extracted_topic])
        else:
            assert isinstance(self.model, KeyedVectors)
            words = extracted_topic.split()
            word_vecs = []
            for word in words:
                if word in self.model:
                    word_vecs.append(self.model[word])
            if word_vecs:
                topic_embedding = np.mean(word_vecs, axis=0).reshape(1, -1)
            else:
                return [("uncategorized", 0.0)]

        assert topic_embedding is not None
        assert self.reference_embeddings is not None, (
            "Reference embeddings not calculated."
        )

        if topic_embedding.ndim == 1:
            topic_embedding = topic_embedding.reshape(1, -1)

        similarities = cosine_similarity(topic_embedding, self.reference_embeddings)[0]

        top_indices = similarities.argsort()[-top_k:][::-1]

        assert self.reference_topics is not None, "Reference topics not set up."
        matches = [(self.reference_topics[i], similarities[i]) for i in top_indices]

        return matches


# Integration with LDA pipeline
def extract_and_map_topics(content, topic_mapper):
    """
    Extract topics using LDA and map to reference topics.

    Returns:
    --------
    dict : Mapped topics with categories and scores
    """
    # Extract topics using existing LDA function
    lda_results = extract_topics_lda(
        content, n_topics=5, preprocessing_pipeline="standard"
    )

    mapped_topics = {"generic": [], "specific": []}

    # Map generic topics
    for topic_words, lda_score in lda_results["generic_topics"]:
        # Create topic string from top words
        topic_string = " ".join([word for word, _ in topic_words[:3]])

        # Map to reference topic
        matches = topic_mapper.map_to_reference_topic(topic_string, top_k=1)
        if matches:
            ref_topic, similarity = matches[0]
            if similarity > 0.3:  # Similarity threshold
                mapped_topics["generic"].append({
                    "reference_topic": ref_topic,
                    "original_words": topic_string,
                    "similarity": similarity,
                    "lda_score": lda_score,
                })

    # Map specific topics
    for topic_words, lda_score in lda_results["specific_topics"]:
        topic_string = " ".join([word for word, _ in topic_words[:3]])
        matches = topic_mapper.map_to_reference_topic(topic_string, top_k=1)
        if matches:
            ref_topic, similarity = matches[0]
            if similarity > 0.3:
                mapped_topics["specific"].append({
                    "reference_topic": ref_topic,
                    "original_words": topic_string,
                    "similarity": similarity,
                    "lda_score": lda_score,
                })

    return mapped_topics


# Usage example
def process_blog_with_mapping(blog_content):
    """
    Process a single blog and return standardized topics.
    """
    # Initialize mapper (do this once for all documents)
    mapper = TopicMapper("sentence-bert")
    mapper.setup_reference_topics()

    # Extract and map topics
    results = extract_and_map_topics(blog_content, mapper)

    # Simplified output for aggregation
    standardized_topics = []

    for category in ["generic", "specific"]:
        for topic_info in results[category]:
            standardized_topics.append({
                "category": category,
                "topic": topic_info["reference_topic"],
                "confidence": topic_info["similarity"] * topic_info["lda_score"],
            })

    return standardized_topics


# For batch processing with consistent topics
class BlogTopicAnalyzer:
    """
    Analyze multiple blogs with consistent topic mapping.
    """

    def __init__(self):
        self.mapper = TopicMapper("sentence-bert")
        self.mapper.setup_reference_topics()
        self.topic_counts = {}

    def process_blogs(self, blog_contents):
        """
        Process multiple blogs and aggregate topics.
        """
        for content in blog_contents:
            topics = extract_and_map_topics(content, self.mapper)

            # Aggregate counts
            for category in ["generic", "specific"]:
                for topic_info in topics[category]:
                    key = (category, topic_info["reference_topic"])
                    if key not in self.topic_counts:
                        self.topic_counts[key] = {"count": 0, "total_confidence": 0}
                    self.topic_counts[key]["count"] += 1
                    self.topic_counts[key]["total_confidence"] += (
                        topic_info["similarity"] * topic_info["lda_score"]
                    )

    def get_top_topics(self, category="all", top_n=2):
        """
        Get top topics by category.
        """
        # Filter by category
        if category == "all":
            filtered_topics = self.topic_counts
        else:
            filtered_topics = {
                k: v for k, v in self.topic_counts.items() if k[0] == category
            }

        # Sort by weighted count (count * average confidence)
        sorted_topics = sorted(
            filtered_topics.items(),
            key=lambda x: x[1]["count"] * (x[1]["total_confidence"] / x[1]["count"]),
            reverse=True,
        )

        return sorted_topics[:top_n]
