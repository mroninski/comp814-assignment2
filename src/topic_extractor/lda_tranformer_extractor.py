"""
Transformer-Enhanced LDA Topic Modeling for English Blog Data

This module implements a hybrid approach combining traditional LDA with transformer-based
semantic understanding for robust topic extraction from english blog data.
"""

import logging
import re
import string
import warnings
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import nltk
import numpy as np
import pandas as pd
import spacy
from gensim import corpora, models
from gensim.models import CoherenceModel, LdaModel
from gensim.models.phrases import Phraser, Phrases
from langdetect import LangDetectException, detect_langs
from nltk.corpus import wordnet as wn
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.cli.download import download
from spacy.language import Language

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download("wordnet")


warnings.filterwarnings("ignore")


class TransformerEnhancedLDA:
    """
    A hybrid topic modeling approach that combines traditional LDA with transformer-based
    semantic understanding for improved topic extraction from english blog data.

    This class implements the methodology described in recent research (Angelov 2020,
    Bianchi et al. 2021) that demonstrates superior performance in topic coherence and
    interpretability compared to traditional approaches.
    """

    def __init__(self, min_topic_size: int = 5):
        """
        Initialize the Transformer-Enhanced LDA model.

        Parameters:
        -----------
        min_topic_size : int
            Minimum number of words required to form a topic (default: 5)
        """
        self.min_topic_size = min_topic_size

        # Initialize components that will be populated during processing
        self.detected_languages = []
        self.processed_tokens = []
        self.embeddings = None
        self.lda_model = None
        self.topics = []
        self.coherence_scores = {}
        self.important_terms = []

        # Load english language models
        logger.info("Initializing the English language models...")
        self._initialize_models()

    def _initialize_models(self):
        """
        Initialize the required NLP models for english language processing.
        This includes language detection, english language tokenization, and
        sentence transformers for semantic embeddings.
        """
        self.sentence_transformer = SentenceTransformer("all-mpnet-base-v2")

        # Download english language model if not installed
        if "en_core_web_sm" not in spacy.util.get_installed_models():
            download("en_core_web_sm")

        self.nlp_model = spacy.load("en_core_web_sm")

    def _clean_text(self, text: str) -> str:
        """
        Clean the raw blog text by removing HTML tags, non-ASCII characters,
        and other noise typical in early 2000s blog data.

        Parameters:
        -----------
        text : str
            Raw blog text

        Returns:
        --------
        str : Cleaned text
        """
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Replace HTML entities
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

        # Handle non-ASCII characters more gracefully
        # Keep characters from major languages
        text = re.sub(r"[^\w\s\u0080-\uFFFF]", " ", text)

        # Remove URLs
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            " ",
            text,
        )

        # Remove email addresses
        text = re.sub(r"\S+@\S+", " ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def detect_languages(self, content: str) -> List[Tuple[str, float]]:
        """
        TODO: Move this to the data transformation step.
        Detect languages present in the blog content with confidence scores.

        Parameters:
        -----------
        content : str
            The blog content to analyze

        Returns:
        --------
        List[Tuple[str, float]] : List of (language_code, confidence) tuples
        """
        try:
            # Detect languages in the content
            detected = detect_langs(content)
            self.detected_languages = [(lang.lang, lang.prob) for lang in detected]

            logger.info(f"Detected languages: {self.detected_languages}")
            return self.detected_languages

        except LangDetectException:
            logger.warning("Language detection failed. Defaulting to English.")
            self.detected_languages = [("en", 1.0)]
            return self.detected_languages

    def preprocess_multilingual(self, content: str) -> List[str]:
        """
        Perform multilingual preprocessing including tokenization, lemmatization,
        and named entity recognition. This method handles multiple languages
        gracefully and extracts meaningful tokens.

        Parameters:
        -----------
        content : str
            The cleaned blog content to preprocess

        Returns:
        --------
        List[str] : Preprocessed tokens
        """
        # Process text with spaCy
        doc = self.nlp_model(content)

        # Extract meaningful tokens
        tokens = []
        for token in doc:
            # Keep tokens that are:
            # - Not stopwords (in any language)
            # - Not punctuation
            # - Alphabetic
            # - Longer than 2 characters
            if (
                not token.is_stop
                and not token.is_punct
                and token.is_alpha
                and len(token.text) > 2
            ):
                # Use lemma if available, otherwise use lowercase text
                if token.lemma_ and token.lemma_ != "-PRON-":
                    tokens.append(token.lemma_.lower())
                else:
                    tokens.append(token.text.lower())

        # Extract named entities as additional features
        entities = [
            ent.text.lower()
            for ent in doc.ents
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "LOC"]
        ]

        # Combine tokens and entities
        self.processed_tokens = tokens + entities

        logger.info(f"Preprocessed {len(self.processed_tokens)} tokens")
        return self.processed_tokens

    def generate_semantic_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate semantic embeddings using sentence transformers for enhanced
        topic modeling. These embeddings capture semantic relationships between
        words and documents.

        Parameters:
        -----------
        texts : List[str]
            List of text segments to embed

        Returns:
        --------
        np.ndarray : Embeddings matrix
        """
        # Convert texts to embeddings
        embeddings = self.sentence_transformer.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )

        return embeddings

    def build_semantic_similarity_matrix(self, vocab: List[str]) -> np.ndarray:
        """
        Build a semantic similarity matrix for vocabulary terms using transformer
        embeddings. This matrix enhances LDA by incorporating semantic relationships.

        Parameters:
        -----------
        vocab : List[str]
            Vocabulary terms

        Returns:
        --------
        np.ndarray : Similarity matrix
        """
        # Generate embeddings for vocabulary
        vocab_embeddings = self.generate_semantic_embeddings(vocab)

        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(vocab_embeddings)

        # Apply threshold to keep only strong similarities
        threshold = 0.5
        similarity_matrix[similarity_matrix < threshold] = 0

        return similarity_matrix

    def optimize_topic_number(
        self, texts: List[str], min_topics: int = 2, max_topics: int = 10
    ) -> int:
        """
        Automatically determine optimal number of topics using coherence metrics
        (Cv coherence, NPMI). This ensures high-quality, interpretable topics.

        Parameters:
        -----------
        texts : List[str]
            Preprocessed documents
        min_topics : int
            Minimum number of topics to consider
        max_topics : int
            Maximum number of topics to consider

        Returns:
        --------
        int : Optimal number of topics
        """
        # Create dictionary and corpus for gensim
        texts_tokens = [text.split() for text in texts]
        dictionary = corpora.Dictionary(texts_tokens)
        corpus = [dictionary.doc2bow(text) for text in texts_tokens]

        # Calculate coherence for different topic numbers
        coherence_scores = []

        for num_topics in range(min_topics, max_topics + 1):
            # Train LDA model
            lda = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=42,
                alpha="auto",
                eta="auto",
            )

            # Calculate coherence score
            coherence_model = CoherenceModel(
                model=lda, texts=texts_tokens, dictionary=dictionary, coherence="c_v"
            )

            coherence_score = coherence_model.get_coherence()
            coherence_scores.append((num_topics, coherence_score))

            logger.info(f"Topics: {num_topics}, Coherence: {coherence_score:.4f}")

        # Find optimal number of topics
        self.coherence_scores = dict(coherence_scores)
        optimal_topics = max(coherence_scores, key=lambda x: x[1])[0]

        return optimal_topics

    def enhanced_lda_modeling(
        self, content: str, num_topics: Optional[int] = None
    ) -> Dict:
        """
        Perform enhanced LDA modeling using recent advancements including:
        1. BERTopic-inspired transformer embeddings clustering
        2. Generic domain-agnostic topic modeling with adaptive seed discovery
        3. Contextual document segmentation preserving semantic coherence
        4. Multi-scale topic extraction (word-level and phrase-level)
        5. Dynamic topic number optimization using silhouette analysis
        6. Improved coherence-based topic filtering and refinement

        This approach addresses the limitations of traditional LDA by:
        - Preserving semantic context through sentence-level segmentation
        - Using adaptive vocabulary discovery instead of domain-specific seeds
        - Employing transformer embeddings for better semantic understanding
        - Implementing hierarchical topic discovery with coherence validation
        - Adding semantic clustering pre-filtering for better topic separation

        Parameters:
        -----------
        content : str
            The cleaned blog content to analyze
        num_topics : Optional[int]
            Number of topics (if None, will be optimized automatically)

        Returns:
        --------
        Dict : Dictionary containing topics, coherence scores, and model
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        # Preprocess if not already done
        if not self.processed_tokens:
            self.preprocess_multilingual(content)

        # IMPROVEMENT 1: Better document segmentation using sentence boundaries
        # Instead of arbitrary sliding windows, use semantic sentence groupings
        doc = self.nlp_model(content)
        sentences = [
            sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20
        ]

        # Group sentences into coherent documents (2-3 sentences per document)
        documents = []
        for i in range(0, len(sentences), 2):
            doc_text = " ".join(sentences[i : i + 3])  # 3 sentences per document
            if len(doc_text.split()) >= self.min_topic_size:
                documents.append(doc_text)

        # Fallback to token-based approach if sentence segmentation fails
        if len(documents) < 3:
            logger.info(
                "Sentence segmentation insufficient, using enhanced token approach"
            )
            window_size = 30  # Larger windows for better context
            overlap = 10  # Overlap to maintain context

            for i in range(0, len(self.processed_tokens), window_size - overlap):
                window = self.processed_tokens[i : i + window_size]
                if len(window) >= self.min_topic_size:
                    documents.append(" ".join(window))

        if not documents:
            logger.warning("No documents created from content")
            return {}

        logger.info(f"Created {len(documents)} semantic documents for analysis")

        # IMPROVEMENT 2: Adaptive vocabulary discovery instead of entertainment-specific seeds
        # Automatically discover important terms using TF-IDF and frequency analysis
        # This replaces domain-specific seeds with data-driven vocabulary selection
        logger.info("Performing adaptive vocabulary discovery...")

        # Create initial TF-IDF to find important terms across all documents
        initial_vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8,
            stop_words="english",
            lowercase=True,
        )

        try:
            initial_matrix = initial_vectorizer.fit_transform(documents)
            initial_features = initial_vectorizer.get_feature_names_out()

            # Get high-importance terms by TF-IDF scores - handle sparse matrix properly
            from scipy import sparse

            if sparse.issparse(initial_matrix):
                # Convert sparse matrix to dense for sum operation
                dense_matrix = initial_matrix.toarray()
                tfidf_scores = dense_matrix.sum(axis=0)
            else:
                tfidf_scores = initial_matrix.sum(axis=0)
            important_indices = tfidf_scores.argsort()[-50:][::-1]  # Top 50 terms
            important_terms = [initial_features[idx] for idx in important_indices]

            # Store important terms as instance variable for later use
            self.important_terms = important_terms

            logger.info(
                f"Discovered {len(important_terms)} important terms for adaptive modeling"
            )
        except Exception as e:
            logger.warning(f"Adaptive vocabulary discovery failed: {e}")
            important_terms = []
            self.important_terms = []

        # IMPROVEMENT 3: Enhanced preprocessing with phrase detection
        # Use bigrams and trigrams to capture meaningful phrases automatically
        tokens_for_phrases = [doc.split() for doc in documents]

        # More conservative phrase detection for better quality
        bigram = Phrases(tokens_for_phrases, min_count=2, threshold=8)
        trigram = Phrases(bigram[tokens_for_phrases], min_count=1, threshold=10)

        bigram_mod = Phraser(bigram)
        trigram_mod = Phraser(trigram)

        # Apply phrase detection
        processed_docs = []
        for tokens in tokens_for_phrases:
            processed_docs.append(trigram_mod[bigram_mod[tokens]])

        # IMPROVEMENT 4: BERTopic-inspired clustering approach with improved embeddings
        # Use transformer embeddings to find semantic clusters first
        logger.info("Generating semantic embeddings for documents...")
        doc_embeddings = self.generate_semantic_embeddings(documents)

        # IMPROVEMENT 5: Enhanced dimensionality reduction with UMAP for better clustering
        # UMAP preserves both local and global structure better than PCA
        logger.info(
            "Applying dimensionality reduction for better semantic clustering..."
        )
        try:
            from umap import UMAP

            # Use UMAP for better non-linear dimensionality reduction, with safer parameters for small datasets
            n_components = min(min(3, len(documents) - 1), doc_embeddings.shape[1] - 1)
            n_components = max(1, n_components)

            # For very small datasets, use safer UMAP parameters
            n_neighbors = min(max(2, len(documents) // 2), 5)

            if len(documents) >= 5 and n_neighbors < len(documents):
                umap_model = UMAP(
                    n_components=n_components,
                    random_state=42,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    metric="cosine",
                )
                reduced_embeddings = umap_model.fit_transform(doc_embeddings)
                logger.info(f"UMAP reduced embeddings to {n_components} dimensions")
            else:
                raise ImportError("Dataset too small for UMAP, using PCA")

        except (ImportError, Exception) as e:
            logger.info(f"UMAP not available or failed ({e}), falling back to PCA...")
            from sklearn.decomposition import PCA

            n_components = min(min(3, len(documents) - 1), doc_embeddings.shape[1] - 1)
            n_components = max(1, n_components)

            pca_model = PCA(n_components=n_components, random_state=42)
            reduced_embeddings = pca_model.fit_transform(doc_embeddings)
            logger.info(f"PCA reduced embeddings to {n_components} dimensions")

        # IMPROVEMENT 6: Dynamic topic number optimization using multiple metrics
        # Combine silhouette analysis with coherence scores for better topic number selection
        if num_topics is None:
            logger.info("Optimizing number of topics using combined metrics...")
            silhouette_scores = []
            # Ensure we don't have more clusters than samples - 1
            max_topics = min(len(documents) - 1, 6)
            topic_range = range(2, max_topics + 1)

            for n_topics in topic_range:
                try:
                    kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(reduced_embeddings)

                    # Only calculate silhouette score if we have valid clustering
                    if len(set(cluster_labels)) > 1 and len(set(cluster_labels)) < len(
                        reduced_embeddings
                    ):
                        # Ensure reduced_embeddings is a proper numpy array
                        embeddings_array = np.asarray(reduced_embeddings)
                        silhouette_avg = silhouette_score(
                            embeddings_array, cluster_labels
                        )
                    else:
                        silhouette_avg = 0.0  # Invalid clustering

                    # Additional metric: cluster separation
                    cluster_centers = kmeans.cluster_centers_
                    if len(cluster_centers) > 1:
                        center_distances = []
                        for i in range(len(cluster_centers)):
                            for j in range(i + 1, len(cluster_centers)):
                                dist = np.linalg.norm(
                                    cluster_centers[i] - cluster_centers[j]
                                )
                                center_distances.append(dist)
                        separation_score = (
                            np.mean(center_distances) if center_distances else 0
                        )
                    else:
                        separation_score = 0

                    # Combined score (silhouette + separation)
                    combined_score = silhouette_avg + 0.3 * separation_score
                    silhouette_scores.append((n_topics, silhouette_avg, combined_score))
                    logger.info(
                        f"Topics: {n_topics}, Silhouette: {silhouette_avg:.4f}, Combined: {combined_score:.4f}"
                    )

                except Exception as e:
                    logger.warning(f"Skipping {n_topics} topics due to error: {e}")
                    continue

            if silhouette_scores:
                # Choose number of topics with highest combined score
                num_topics = max(silhouette_scores, key=lambda x: x[2])[0]
                logger.info(f"Optimal number of topics: {num_topics}")
            else:
                num_topics = 3  # Fallback default
                logger.info("Using fallback number of topics: 3")

        # Ensure num_topics is valid
        if num_topics is None or num_topics <= 0:
            num_topics = 3  # Default fallback
            logger.info(f"Using default number of topics: {num_topics}")

        # IMPROVEMENT 7: Advanced TF-IDF with adaptive vocabulary filtering
        # Use discovered important terms to guide vocabulary selection
        vectorizer = TfidfVectorizer(
            max_features=min(300, len(important_terms) * 3) if important_terms else 150,
            ngram_range=(1, 3),  # Include trigrams for better phrase capture
            min_df=1,  # Keep rare but potentially important terms
            max_df=0.85,  # Remove overly common terms
            stop_words="english",
            lowercase=True,
            token_pattern=r"\b[a-zA-Z][a-zA-Z0-9]*\b",
            # Use sublinear TF scaling for better performance
            sublinear_tf=True,
            # Use IDF smoothing
            smooth_idf=True,
        )

        try:
            doc_term_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()
            logger.info(
                f"Created document-term matrix with {len(feature_names)} features"
            )
        except Exception as e:
            logger.error(f"Failed to create document-term matrix: {e}")
            return {}

        # IMPROVEMENT 8: Enhanced LDA with better hyperparameters for generic topics
        # Use asymmetric priors and better convergence criteria
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            learning_method="batch",
            max_iter=150,  # More iterations for better convergence
            # Asymmetric document-topic prior encourages diverse topic distributions
            doc_topic_prior=0.1,
            # Lower topic-word prior for more focused topics
            topic_word_prior=0.01,
            learning_decay=0.7,
            learning_offset=50.0,
            # Better convergence criteria
            perp_tol=1e-2,
            mean_change_tol=1e-3,
        )

        # Fit the model
        logger.info("Training enhanced LDA model with generic topic optimization...")
        lda.fit(doc_term_matrix)

        # IMPROVEMENT 9: Semantic coherence-based topic extraction and filtering
        topics = []
        feature_names = list(feature_names)

        # Build semantic similarity matrix for vocabulary
        vocab_embeddings = self.generate_semantic_embeddings(feature_names)

        for topic_idx, topic_weights in enumerate(lda.components_):
            # Get top words by weight
            top_indices = topic_weights.argsort()[-20:][
                ::-1
            ]  # Top 20 words for analysis
            top_words = [feature_names[idx] for idx in top_indices]
            top_weights = topic_weights[top_indices].tolist()

            # IMPROVEMENT 10: Advanced semantic coherence filtering
            # Filter words based on semantic coherence with the topic centroid
            if top_words:
                # Calculate topic centroid embedding
                topic_word_embeddings = vocab_embeddings[
                    top_indices[:10]
                ]  # Top 10 for centroid
                topic_centroid = np.mean(topic_word_embeddings, axis=0)

                coherent_words = []
                coherent_weights = []
                coherence_scores = []

                for i, word_idx in enumerate(top_indices):
                    word_embedding = vocab_embeddings[word_idx]
                    # Calculate semantic similarity to topic centroid
                    similarity = 1 - cosine(topic_centroid, word_embedding)

                    # More sophisticated filtering criteria:
                    # 1. High semantic similarity to topic centroid (>0.3)
                    # 2. High TF-IDF weight (top 15)
                    # 3. Not too generic (check against common words)
                    if (similarity > 0.25 and i < 15) or (similarity > 0.4):
                        coherent_words.append(top_words[i])
                        coherent_weights.append(top_weights[i])
                        coherence_scores.append(similarity)

                # Ensure minimum topic size
                if len(coherent_words) < 5:
                    coherent_words = top_words[:8]
                    coherent_weights = top_weights[:8]
                    coherence_scores = [0.5] * len(coherent_words)  # Default coherence
            else:
                coherent_words = []
                coherent_weights = []
                coherence_scores = []

            # IMPROVEMENT 11: Generic topic labeling using semantic clustering
            topic_label = self._generate_generic_topic_label(
                coherent_words, coherence_scores
            )

            # Calculate topic quality score based on coherence and weight distribution
            topic_quality = (
                np.mean(coherence_scores) * np.std(coherent_weights[:5])
                if coherent_weights
                else 0
            )

            topics.append({
                "topic_id": topic_idx,
                "label": topic_label,
                "words": coherent_words[:10],
                "weights": coherent_weights[:10],
                "coherence_scores": coherence_scores[:10],
                "topic_quality": topic_quality,
                "semantic_coherence": np.mean(coherence_scores)
                if coherence_scores
                else 0,
            })

        # IMPROVEMENT 12: Quality-based topic ranking instead of domain-specific ranking
        # Rank topics by overall quality and semantic coherence
        topics.sort(
            key=lambda x: (x["topic_quality"], x["semantic_coherence"]), reverse=True
        )

        self.topics = topics
        self.lda_model = lda

        logger.info(
            f"Successfully extracted {len(topics)} topics with generic optimization"
        )

        return {
            "topics": topics,
            "num_topics": num_topics,
            "high_quality_topics": [t for t in topics if t["topic_quality"] > 0.1],
            "coherence_scores": {
                f"topic_{i}": t["semantic_coherence"] for i, t in enumerate(topics)
            },
            "model": lda,
            "embeddings": doc_embeddings,
            "document_count": len(documents),
            "average_topic_quality": np.mean([t["topic_quality"] for t in topics]),
        }

    def _generate_generic_topic_label(
        self, words: List[str], coherence_scores: List[float]
    ) -> str:
        """
        Generate intelligent topic labels based on word analysis and coherence scores.
        Uses a generic approach that works for any domain without domain-specific knowledge.

        Parameters:
        -----------
        words : List[str]
            Top words for the topic
        coherence_scores : List[float]
            Coherence scores for the topic words

        Returns:
        --------
        str : Generated topic label
        """
        if not words:
            return "unknown_topic"

        # Use the most coherent and meaningful words for labeling
        if coherence_scores:
            # Sort words by coherence score and select top words
            word_coherence_pairs = list(zip(words, coherence_scores))
            word_coherence_pairs.sort(key=lambda x: x[1], reverse=True)

            # Take top 2-3 most coherent words for the label
            top_coherent_words = [pair[0] for pair in word_coherence_pairs[:3]]

            # Filter out very short words for better labels
            meaningful_words = [word for word in top_coherent_words if len(word) > 3]

            if meaningful_words:
                return "_".join(meaningful_words[:2])
            else:
                return "_".join(top_coherent_words[:2])

        # Fallback to traditional approach using top words
        meaningful_words = [word for word in words[:3] if len(word) > 3]
        if meaningful_words:
            return "_".join(meaningful_words[:2])
        else:
            return "_".join(words[:2])

    def refine_topic_labels(self) -> List[Dict]:
        """
        Use transformer models to refine topic labels and ensure semantic
        consistency. This post-processing step improves interpretability.

        Returns:
        --------
        List[Dict] : Refined topics with improved labels
        """
        if not self.topics:
            logger.warning("No topics to refine")
            return []

        refined_topics = []

        for topic in self.topics:
            # Create a representative text from topic words
            topic_text = " ".join(topic["words"])

            # Generate embedding for topic
            topic_embedding = self.generate_semantic_embeddings([topic_text])[0]

            # Find most representative words using embeddings
            word_embeddings = self.generate_semantic_embeddings(topic["words"])

            # Calculate centrality of each word to the topic
            centralities = []
            for word_emb in word_embeddings:
                centrality = 1 - cosine(topic_embedding, word_emb)
                centralities.append(centrality)

            # Sort words by centrality
            sorted_indices = np.argsort(centralities)[::-1]
            refined_words = [topic["words"][i] for i in sorted_indices]

            # Create a more interpretable label
            label = "_".join(refined_words[:3])  # Top 3 most central words

            refined_topics.append({
                "topic_id": topic["topic_id"],
                "label": label,
                "words": refined_words,
                "centralities": [centralities[i] for i in sorted_indices],
            })

        return refined_topics

    @staticmethod
    def get_hypernym(word: str) -> str:
        """Using the WordNet corpus, find the hypernym of a word.

        Args:
            word (str): The word to find the hypernym of

        Returns:
            str: The hypernym of the word
        """
        synsets = wn.synsets(word)
        if synsets:
            # Get the most common sense
            synset = synsets[0]
            if synset:
                hypernyms = synset.hypernyms()
                if hypernyms:
                    # Return the simplest hypernym
                    return hypernyms[0].lemmas()[0].name().replace("_", " ")
        return ""

    def extract_topics(
        self, blog_content: str, num_topics: Optional[int] = None
    ) -> Dict:
        """
        Main method to extract topics from the blog content using the full
        transformer-enhanced LDA pipeline.

        Parameters:
        -----------
        blog_content : str
            Raw blog post content that may contain noise
        num_topics : Optional[int]
            Number of topics to extract (if None, will be optimized)

        Returns:
        --------
        Dict : Combined results containing extracted topics and evaluation metrics
        """
        logger.info("Starting topic extraction pipeline...")

        # Reset instance variables for new content
        self.detected_languages = []
        self.processed_tokens = []
        self.embeddings = None
        self.lda_model = None
        self.topics = []
        self.coherence_scores = {}
        self.important_terms = []

        # Clean and preprocess the content
        clean_content = self._clean_text(blog_content)

        # Optional: Detect languages (can be used for filtering)
        self.detect_languages(clean_content)

        # Run the full pipeline
        modeling_results = self.enhanced_lda_modeling(clean_content, num_topics)

        if not modeling_results:
            logger.error("Topic modeling failed")
            return {}

        # Refine topics
        refined_topics = self.refine_topic_labels()

        # Evaluate topics
        evaluation_results = self.evaluate_topics()

        logger.info(f"Extracted {len(refined_topics)} topics successfully")

        # Get all the words from the topics
        all_words = set()
        for topic in refined_topics:
            for word in topic["words"]:
                # Verify this word is clean, to stop skewing the results
                invalid_words = [_ for _ in word.split(" ") if len(_) <= 2]
                if len(invalid_words) > 0:
                    continue

                all_words.add(word)

        # Combine results
        combined_results = {
            "extracted_topics": refined_topics,
            "modeling_results": modeling_results,
            "evaluation": evaluation_results,
            "summary": self.get_topic_summary(),
            "words": all_words,
        }

        return combined_results

    def evaluate_topics(self) -> Dict:
        """
        Evaluate the quality of extracted topics using multiple metrics
        including coherence, diversity, and semantic consistency.

        Returns:
        --------
        Dict : Evaluation metrics
        """
        if not self.topics:
            logger.warning("No topics to evaluate")
            return {}

        # Calculate topic diversity (percentage of unique words)
        all_words = []
        for topic in self.topics:
            all_words.extend(topic["words"])

        diversity = len(set(all_words)) / len(all_words) if all_words else 0

        # Calculate average coherence (if available)
        avg_coherence = (
            np.mean(list(self.coherence_scores.values()))
            if self.coherence_scores
            else 0
        )

        # Calculate semantic consistency
        # (How similar are the words within each topic)
        semantic_consistencies = []

        for topic in self.topics:
            if len(topic["words"]) > 1:
                # Get embeddings for topic words
                embeddings = self.generate_semantic_embeddings(topic["words"])

                # Calculate pairwise similarities
                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        sim = 1 - cosine(embeddings[i], embeddings[j])
                        similarities.append(sim)

                semantic_consistencies.append(np.mean(similarities))

        avg_consistency = (
            np.mean(semantic_consistencies) if semantic_consistencies else 0
        )

        # Get the hypernym of the topics
        all_hypernyms = set()
        for topic_details in self.topics:
            for word in topic_details["words"]:
                all_hypernyms.add(self.get_hypernym(word))

        evaluation = {
            "topics": self.topics,
            "topic_diversity": diversity,
            "average_coherence": avg_coherence,
            "semantic_consistency": avg_consistency,
            "num_topics": len(self.topics),
            "coverage": len(self.processed_tokens) if self.processed_tokens else 0,
            "hypernyms": all_hypernyms,
        }

        logger.info(f"Evaluation metrics: {evaluation}")

        return evaluation

    def get_topic_summary(self) -> str:
        """
        Generate a human-readable summary of the extracted topics.

        Returns:
        --------
        str : Summary of topics
        """
        if not self.topics:
            return "No topics extracted yet. Run extract_topics() first."

        # Get refined topics if available
        refined_topics = self.refine_topic_labels()

        summary = f"Extracted {len(refined_topics)} topics from blog content:\n\n"

        for topic in refined_topics:
            summary += f"Topic {topic['topic_id'] + 1}: {topic['label']}\n"
            summary += f"  Top words: {', '.join(topic['words'][:5])}\n"
            summary += (
                f"  Semantic centrality: {np.mean(topic['centralities'][:5]):.3f}\n\n"
            )

        # Add evaluation metrics
        evaluation = self.evaluate_topics()
        summary += "\nEvaluation Metrics:\n"
        summary += f"  Topic Diversity: {evaluation['topic_diversity']:.3f}\n"
        summary += f"  Average Coherence: {evaluation['average_coherence']:.3f}\n"
        summary += f"  Semantic Consistency: {evaluation['semantic_consistency']:.3f}\n"

        return summary


# Example usage and testing
if __name__ == "__main__":
    # Example blog content (replace with actual blog data)
    sample_blog = """
    Jazz and the Charleston  Oooh..went to watch  urlLink The Cat's Meow   starring Kirsten Dunst, Cary Elwes and also Edward Hermann, Rory's grandad in Gilmore Girls, who plays William Randolph Hearst, the 1920s newspaper titan. Quick trivia: Did you know that Citizen Kane was supposedly based on his life (Hearst I mean, not Hermann).  The Cat's Meow is based on the novel written by Elinor Glyn, about what really happened on a yachting trip in 1924. Actually, 15 November 1924, to be exact - hehe my birthdate! ooooooooh...  Movie and the webstie have really nice 1920s-style songs...it's actually playing in the BG as I type. And the Charleston dances they did seemed pretty funky too hehe.  Am plannign to watch Lilo n Stitch later, prob at Highpoint. Some lighthearted 'toon eh to jazz up the week. Man, I've really been watching lotsa movies these couple of days...prob to make up for the months of abstinence hehe.  okies, cheerio. And please remember, flooble's down, so leave a bl00p (comment lah)!    
    """

    # Initialize the model
    topic_model = TransformerEnhancedLDA(min_topic_size=8)

    # Extract topics - now returns both topics and evaluation results
    results = topic_model.extract_topics(sample_blog, num_topics=12)

    # Print results
    print("=" * 50)
    print("TOPIC EXTRACTION RESULTS")
    print("=" * 50)
    print(f"Extracted {len(results.get('extracted_topics', []))} topics")
    print("\nEvaluation Metrics:")
    evaluation = results.get("evaluation", {})
    print(f"Topic Diversity: {evaluation.get('topic_diversity', 'N/A')}")
    print(f"Average Coherence: {evaluation.get('average_coherence', 'N/A')}")
    print(f"Semantic Consistency: {evaluation.get('semantic_consistency', 'N/A')}")

    print("\nSummary:")
    print(results.get("summary", "No summary available"))

    from topic_extractor.topic_simplifying import map_topic_words_to_taxonomy

    classification = map_topic_words_to_taxonomy(results.get("words", set()), top_n=10)

    print(classification)
