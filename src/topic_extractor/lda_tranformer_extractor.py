"""
Transformer-Enhanced LDA Topic Modeling for Blog Data Analysis

A streamlined implementation combining LDA with transformer-based semantic understanding
for robust topic extraction from blog content. Optimized for the full_analysis_dataset pipeline.
"""

import logging
import re
import warnings
from typing import Dict, List, Optional

import nltk
import numpy as np
import spacy
from gensim.models.phrases import Phraser, Phrases
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.cli.download import download
from umap import UMAP

# Configure logging for production use
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Download required NLTK data silently
try:
    nltk.data.find("wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

warnings.filterwarnings("ignore")


class TransformerEnhancedLDA:
    """
    Streamlined topic modeling implementation combining LDA with transformer embeddings.

    This class focuses on robust topic extraction from blog content for downstream analysis.
    Key features:
    - Semantic document segmentation using spaCy sentence boundaries
    - Transformer embeddings for improved semantic understanding
    - Adaptive topic number optimization using silhouette analysis
    - Quality-based topic filtering for meaningful results
    """

    def __init__(self, min_topic_size: int = 5):
        """
        Initialize the topic extraction model.

        Args:
            min_topic_size: Minimum number of words required to form a meaningful topic
        """
        self.min_topic_size = min_topic_size
        self._initialize_models()

    def _initialize_models(self):
        """Initialize required NLP models for processing."""
        # Load transformer model for semantic embeddings
        self.sentence_transformer = SentenceTransformer("all-mpnet-base-v2")

        # Download and load spaCy English model if not available
        if "en_core_web_sm" not in spacy.util.get_installed_models():
            download("en_core_web_sm")
        self.nlp_model = spacy.load("en_core_web_sm")

        # Changing the max length to a very large number to avoid errors
        self.nlp_model.max_length = 100000000000000

    def _clean_text(self, text: str) -> str:
        """
        Clean raw blog text by removing HTML artifacts and noise.

        Args:
            text: Raw blog content

        Returns:
            Cleaned text suitable for topic modeling
        """
        # Remove HTML tags and entities
        text = re.sub(r"<[^>]+>", " ", text)
        html_entities = {
            "&nbsp;": " ",
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
            "&quot;": '"',
            "&#39;": "'",
            "&ndash;": "-",
            "&mdash;": "-",
        }
        for entity, replacement in html_entities.items():
            text = text.replace(entity, replacement)

        # Remove URLs and email addresses
        text = re.sub(r"http[s]?://\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _preprocess_text(self, content: str) -> List[str]:
        """
        Extract meaningful tokens from content using spaCy processing.

        Args:
            content: Cleaned blog content

        Returns:
            List of preprocessed tokens
        """
        doc = self.nlp_model(content)

        tokens = []
        for token in doc:
            # Keep meaningful tokens: not stopwords, not punctuation, alphabetic, length > 2
            if (
                not token.is_stop
                and not token.is_punct
                and token.is_alpha
                and len(token.text) > 2
            ):
                # Use lemma for better generalization
                lemma = token.lemma_ if token.lemma_ != "-PRON-" else token.text
                tokens.append(lemma.lower())

        # Extract named entities as additional features
        entities = [
            ent.text.lower()
            for ent in doc.ents
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "LOC"]
        ]

        return tokens + entities

    def _create_semantic_documents(self, content: str) -> List[str]:
        """
        Create semantic documents using sentence boundaries for better topic coherence.

        Args:
            content: Cleaned blog content

        Returns:
            List of semantic documents for topic modeling
        """
        doc = self.nlp_model(content)
        sentences = [
            sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20
        ]

        # Group sentences into coherent documents (2-3 sentences each)
        documents = []
        for i in range(0, len(sentences), 2):
            doc_text = " ".join(sentences[i : i + 3])
            if len(doc_text.split()) >= self.min_topic_size:
                documents.append(doc_text)

        # Fallback to token-based windowing if insufficient documents
        if len(documents) < 3:
            tokens = self._preprocess_text(content)
            window_size, overlap = 30, 10

            for i in range(0, len(tokens), window_size - overlap):
                window = tokens[i : i + window_size]
                if len(window) >= self.min_topic_size:
                    documents.append(" ".join(window))

        return documents

    def _optimize_topic_number(
        self, embeddings: np.ndarray, max_topics: int = 6
    ) -> int:
        """
        Determine optimal number of topics using silhouette analysis.

        Args:
            embeddings: Document embeddings for clustering analysis
            max_topics: Maximum number of topics to consider

        Returns:
            Optimal number of topics
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        max_clusters = min(len(embeddings) - 1, max_topics)
        best_score, best_topics = -1, 3

        for n_topics in range(2, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)

                if len(set(labels)) > 1:
                    score = silhouette_score(embeddings, labels)
                    if score > best_score:
                        best_score, best_topics = score, n_topics
            except:
                continue

        return best_topics

    def _enhance_vocabulary_with_phrases(self, documents: List[str]) -> List[str]:
        """
        Enhance vocabulary by detecting meaningful phrases using gensim.

        Args:
            documents: List of documents for phrase detection

        Returns:
            Documents with detected phrases included
        """
        # Convert to token lists for phrase detection
        token_docs = [doc.split() for doc in documents]

        # Build phrase models with conservative parameters
        bigram = Phrases(token_docs, min_count=2, threshold=8)
        trigram = Phrases(bigram[token_docs], min_count=1, threshold=10)

        bigram_mod = Phraser(bigram)
        trigram_mod = Phraser(trigram)

        # Apply phrase detection and return as text
        enhanced_docs = []
        for tokens in token_docs:
            enhanced_tokens = trigram_mod[bigram_mod[tokens]]
            enhanced_docs.append(" ".join(enhanced_tokens))

        return enhanced_docs

    def _calculate_topic_quality(
        self, words: List[str], weights: List[float], coherence_scores: List[float]
    ) -> float:
        """
        Calculate comprehensive topic quality score for filtering.

        Args:
            words: Topic words
            weights: Corresponding TF-IDF weights
            coherence_scores: Semantic coherence scores

        Returns:
            Quality score between 0 and 1
        """
        if not words or not weights or not coherence_scores:
            return 0.0

        # Weighted semantic coherence (primary factor - 50%)
        top_coherence = sorted(coherence_scores, reverse=True)[:3]
        if len(top_coherence) >= 3:
            weighted_coherence = (
                0.5 * top_coherence[0] + 0.3 * top_coherence[1] + 0.2 * top_coherence[2]
            )
        else:
            weighted_coherence = np.mean(coherence_scores)
        coherence_component = min(1.0, float(weighted_coherence))

        # Word distinctiveness via weight distribution (30%)
        weights_array = np.array(weights[:8])
        if len(weights_array) > 1 and np.sum(weights_array) > 0:
            probs = weights_array / np.sum(weights_array)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(len(probs))
            distinctiveness = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
        else:
            distinctiveness = 0.0

        # Vocabulary diversity (20%)
        unique_stems = set(word[:4] for word in words[:6] if len(word) > 3)
        vocab_richness = min(1.0, len(unique_stems) / 4.0)

        # Composite score
        return float(
            0.5 * coherence_component + 0.3 * distinctiveness + 0.2 * vocab_richness
        )

    def _generate_topic_label(self, words: List[str]) -> str:
        """
        Generate concise topic labels from top words.

        Args:
            words: Top words for the topic

        Returns:
            Descriptive topic label
        """
        if not words:
            return "unknown_topic"

        # Use top 2-3 meaningful words (length > 3) for label
        meaningful_words = [word for word in words[:4] if len(word) > 3]
        if meaningful_words:
            return "_".join(meaningful_words[:2])
        else:
            return "_".join(words[:2])

    def _filter_meaningful_topics(self, topics: List[Dict]) -> List[Dict]:
        """
        Filter topics to retain only the most meaningful ones.

        Args:
            topics: List of extracted topics with quality scores

        Returns:
            Filtered list of high-quality topics
        """
        if not topics:
            return []

        qualities = [t["topic_quality"] for t in topics]
        max_quality = max(qualities)
        mean_quality = np.mean(qualities)

        # Adaptive quality threshold
        if max_quality > 0.3:
            threshold = 0.7 * max_quality
        else:
            threshold = max(0.8 * max_quality, mean_quality)
        threshold = max(threshold, 0.01)

        # Filter by quality and remove duplicates
        meaningful_topics = []
        for topic in sorted(topics, key=lambda x: x["topic_quality"], reverse=True):
            if topic["topic_quality"] >= threshold:
                # Check for word overlap with existing topics
                topic_words = set(topic["words"][:5])
                is_unique = True

                for existing in meaningful_topics:
                    existing_words = set(existing["words"][:5])
                    overlap_ratio = len(topic_words & existing_words) / min(
                        len(topic_words), len(existing_words)
                    )
                    if overlap_ratio > 0.6:
                        is_unique = False
                        break

                if is_unique:
                    meaningful_topics.append(topic)

        return meaningful_topics[:3]  # Limit to top 3 topics

    def extract_topics(
        self, blog_content: str, num_topics: Optional[int] = None, num_words: int = 10
    ) -> Dict:
        """
        Extract topics from blog content using transformer-enhanced LDA.

        This is the main entry point for topic extraction, optimized for the
        full_analysis_dataset pipeline.

        Args:
            blog_content: Raw blog content to analyze
            num_topics: Number of topics to extract (auto-optimized if None)
            num_words: Number of words per topic to return

        Returns:
            Dictionary containing:
            - lda_results: Full topic modeling results with quality metrics
            - words: Set of all significant topic words
            - topic_labels: List of topic labels for downstream processing
        """
        # Step 1: Clean and preprocess content
        clean_content = self._clean_text(blog_content)
        documents = self._create_semantic_documents(clean_content)

        if len(documents) < 2:
            logger.warning("Insufficient content for topic modeling")
            return {"lda_results": {"topics": []}, "words": set(), "topic_labels": []}

        # Step 2: Enhance documents with phrase detection
        enhanced_documents = self._enhance_vocabulary_with_phrases(documents)

        # Step 3: Generate semantic embeddings for topic optimization
        doc_embeddings = self.sentence_transformer.encode(
            enhanced_documents, convert_to_numpy=True
        )

        # Step 4: Optimize number of topics if not specified
        if num_topics is None:
            num_topics = self._optimize_topic_number(doc_embeddings)

        # Step 5: Apply dimensionality reduction for better clustering
        n_components = min(3, len(documents) - 1, doc_embeddings.shape[1] - 1)
        n_components = max(1, n_components)

        try:
            if len(documents) >= 5:
                n_neighbors = min(5, len(documents) // 2)
                umap_model = UMAP(
                    n_components=n_components,
                    random_state=42,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    metric="cosine",
                )
                reduced_embeddings = umap_model.fit_transform(doc_embeddings)
            else:
                raise ImportError("Dataset too small for UMAP")
        except:
            pca_model = PCA(n_components=n_components, random_state=42)
            reduced_embeddings = pca_model.fit_transform(doc_embeddings)

        # Step 6: Create document-term matrix
        vectorizer = TfidfVectorizer(
            max_features=150,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.85,
            stop_words="english",
            lowercase=True,
            sublinear_tf=True,
            smooth_idf=True,
        )

        try:
            doc_term_matrix = vectorizer.fit_transform(enhanced_documents)
            feature_names = vectorizer.get_feature_names_out()
        except Exception as e:
            logger.error(f"Failed to create document-term matrix: {e}")
            return {"lda_results": {"topics": []}, "words": set(), "topic_labels": []}

        # Step 7: Train LDA model with optimized parameters
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            learning_method="batch",
            max_iter=150,
            doc_topic_prior=0.1,
            topic_word_prior=0.01,
            learning_decay=0.7,
            learning_offset=50.0,
            perp_tol=1e-2,
            mean_change_tol=1e-3,
        )
        lda.fit(doc_term_matrix)

        # Step 8: Extract and evaluate topics
        vocab_embeddings = self.sentence_transformer.encode(
            list(feature_names), convert_to_numpy=True
        )
        topics = []

        for topic_idx, topic_weights in enumerate(lda.components_):
            # Get top words by LDA weight
            top_indices = topic_weights.argsort()[-20:][::-1]
            top_words = [str(feature_names[idx]) for idx in top_indices]
            top_weights = topic_weights[top_indices].tolist()

            # Calculate semantic coherence with topic centroid
            if len(top_indices) > 0:
                topic_centroid = np.mean(vocab_embeddings[top_indices[:8]], axis=0)
                coherence_scores = []
                coherent_words, coherent_weights = [], []

                for i, word_idx in enumerate(top_indices):
                    word_embedding = vocab_embeddings[word_idx]
                    similarity = 1 - cosine(topic_centroid, word_embedding)

                    # Filter by semantic coherence and position
                    if (similarity > 0.25 and i < 15) or similarity > 0.35:
                        coherent_words.append(top_words[i])
                        coherent_weights.append(top_weights[i])
                        coherence_scores.append(similarity)

                # Ensure minimum topic size
                if len(coherent_words) < 5:
                    coherent_words = top_words[:8]
                    coherent_weights = top_weights[:8]
                    coherence_scores = [0.5] * len(coherent_words)
            else:
                coherent_words, coherent_weights, coherence_scores = [], [], []

            # Calculate topic quality and create topic dictionary
            topic_quality = self._calculate_topic_quality(
                coherent_words, coherent_weights, coherence_scores
            )
            topic_label = self._generate_topic_label(coherent_words)

            topics.append({
                "topic_id": topic_idx,
                "label": topic_label,
                "words": coherent_words[:num_words],
                "weights": coherent_weights[:num_words],
                "coherence_scores": coherence_scores[:num_words],
                "topic_quality": topic_quality,
                "semantic_coherence": np.mean(coherence_scores)
                if coherence_scores
                else 0,
            })

        # Step 9: Filter meaningful topics and prepare output
        meaningful_topics = self._filter_meaningful_topics(topics)

        # Collect all significant words and topic labels
        all_words = set()
        topic_labels = []

        for topic in meaningful_topics:
            # Add valid words (filter out invalid entries)
            for word in topic["words"]:
                if (
                    isinstance(word, str)
                    and len(word.strip()) > 2
                    and not any(len(w) <= 2 for w in word.split())
                ):
                    all_words.add(word)

            # Process topic labels
            label_parts = topic["label"].split("_")
            topic_labels.extend([part for part in label_parts if len(part) > 2])

        # Prepare final results
        lda_results = {
            "topics": meaningful_topics,
            "num_topics": len(meaningful_topics),
            "document_count": len(documents),
            "average_topic_quality": np.mean([
                t["topic_quality"] for t in meaningful_topics
            ])
            if meaningful_topics
            else 0,
        }

        return {
            "lda_results": lda_results,
            "words": all_words,
            "topic_labels": topic_labels,
        }


# Example usage for testing
if __name__ == "__main__":
    import json

    sample_blog = """
    Machine learning and artificial intelligence are transforming how we process data.
    Natural language processing enables computers to understand human language.
    Deep learning algorithms use neural networks to identify patterns in large datasets.
    These technologies are being applied in healthcare, finance, and autonomous vehicles.
    """

    topic_model = TransformerEnhancedLDA(min_topic_size=2)
    results = topic_model.extract_topics(sample_blog, num_topics=3)

    print("REFACTORED TOPIC EXTRACTION RESULTS")
    print("=" * 50)
    print(f"Extracted {len(results.get('lda_results', {}).get('topics', []))} topics")
    print(json.dumps(results, indent=2, default=str))
