"""
Transformer-Enhanced LDA Topic Modeling with Multilingual Support

This module implements a hybrid approach combining traditional LDA with transformer-based
semantic understanding for robust topic extraction from multilingual blog data.
"""

import logging

# Text preprocessing
import re
import string
import warnings
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Multilingual NLP processing
import spacy

# Topic modeling and NLP utilities
from gensim import corpora, models
from gensim.models import CoherenceModel, LdaModel
from gensim.models.phrases import Phraser, Phrases
from langdetect import LangDetectException, detect_langs

# For coherence optimization
from scipy.spatial.distance import cosine

# Sentence transformers for semantic embeddings
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation

# Scientific computing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.language import Language

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from nltk.corpus import wordnet as wn
import nltk
nltk.download('wordnet')


warnings.filterwarnings("ignore")

# Download all spacy models
from spacy.cli import download

LANGUAGES_MAP = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "es": "es_core_news_sm",
    "fr": "fr_core_news_sm",
    "it": "it_core_news_sm",
    "pt": "pt_core_news_sm",
    "ru": "ru_core_news_sm",
    "sv": "sv_core_news_sm",
    "zh": "zh_core_web_sm",
    "ja": "ja_core_news_sm",
    "ko": "ko_core_news_sm",
    "ar": "ar_core_news_sm",
    "hi": "hi_core_news_sm",
    "bn": "bn_core_news_sm",
    "mr": "mr_core_news_sm",
    "ta": "ta_core_news_sm",
}




class TransformerEnhancedLDA:
    """
    A hybrid topic modeling approach that combines traditional LDA with transformer-based
    semantic understanding for improved topic extraction from multilingual blog data.

    This class implements the methodology described in recent research (Angelov 2020,
    Bianchi et al. 2021) that demonstrates superior performance in topic coherence and
    interpretability compared to traditional approaches.
    """

    def __init__(self, blog_content: str, min_topic_size: int = 5):
        """
        Initialize the Transformer-Enhanced LDA model with blog content.

        Parameters:
        -----------
        blog_content : str
            Raw blog post content that may contain multiple languages and noise
        min_topic_size : int
            Minimum number of words required to form a topic (default: 5)
        """
        self.raw_content = blog_content
        self.min_topic_size = min_topic_size

        # Initialize components
        self.detected_languages = []
        self.processed_tokens = []
        self.embeddings = None
        self.lda_model = None
        self.topics = []
        self.coherence_scores = {}
        
        # Clean and preprocess the content
        self.clean_content = self._clean_text(blog_content)

        # Load multilingual models
        logger.info("Initializing multilingual models...")
        self.detect_languages()
        self._initialize_models()

    def _initialize_models(self):
        """
        Initialize the required NLP models for multilingual processing.
        This includes language detection, multilingual tokenization, and
        sentence transformers for semantic embeddings.
        """
        try:
            # Initialize multilingual sentence transformer
            # Using paraphrase-multilingual-MiniLM-L12-v2 for good balance of speed and quality
            self.sentence_transformer = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2"
            )

            # Initialize spaCy models for major languages
            # We'll use a lightweight multilingual model
            self.nlp_models = {}

            # Try to load multilingual model, fallback to English if not available
            self.nlp_models = {}
            try:
                for lang_details in self.detected_languages:
                    lang, _ = lang_details
                    model_name = LANGUAGES_MAP[lang]
                    self.nlp_models[lang] = spacy.load(model_name)
            except OSError as e:
                logger.error(f"Likely missing spaCy model: {model_name}")
                download(model_name)
                self.nlp_models[lang] = spacy.load(model_name)
                logger.info(f"Downloaded spaCy model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

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

    def detect_languages(self) -> List[Tuple[str, float]]:
        """
        Detect languages present in the blog content with confidence scores.

        Returns:
        --------
        List[Tuple[str, float]] : List of (language_code, confidence) tuples
        """
        try:
            # Detect languages in the content
            detected = detect_langs(self.clean_content)
            self.detected_languages = [(lang.lang, lang.prob) for lang in detected]

            logger.info(f"Detected languages: {self.detected_languages}")
            return self.detected_languages

        except LangDetectException:
            logger.warning("Language detection failed. Defaulting to English.")
            self.detected_languages = [("en", 1.0)]
            return self.detected_languages

    def preprocess_multilingual(self) -> List[str]:
        """
        Perform multilingual preprocessing including tokenization, lemmatization,
        and named entity recognition. This method handles multiple languages
        gracefully and extracts meaningful tokens.

        Returns:
        --------
        List[str] : Preprocessed tokens
        """
        # Detect languages first
        if not self.detected_languages:
            self.detect_languages()

        # Use appropriate spaCy model based on detected language
        primary_lang = (
            self.detected_languages[0][0] if self.detected_languages else "en"
        )

        # Select appropriate model
        if "multi" in self.nlp_models:
            nlp = self.nlp_models["multi"]
        elif primary_lang[:2] in self.nlp_models:
            nlp = self.nlp_models[primary_lang[:2]]
        else:
            nlp = self.nlp_models.get("en", None)

        if nlp is None:
            logger.error("No NLP model available for preprocessing")
            return []

        # Process text with spaCy
        doc = nlp(self.clean_content)

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
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT"]
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

    def enhanced_lda_modeling(self, num_topics: Optional[int] = None) -> Dict:
        """
        Perform enhanced LDA modeling that incorporates semantic similarity
        matrices from transformer embeddings. This hybrid approach combines
        the interpretability of LDA with semantic understanding.

        Parameters:
        -----------
        num_topics : Optional[int]
            Number of topics (if None, will be optimized automatically)

        Returns:
        --------
        Dict : Dictionary containing topics, coherence scores, and model
        """
        # Preprocess if not already done
        if not self.processed_tokens:
            self.preprocess_multilingual()

        # Create documents from tokens (group into sentences or chunks)
        # For blog data, we'll use sliding windows
        window_size = 20
        documents = []

        for i in range(0, len(self.processed_tokens), window_size // 2):
            window = self.processed_tokens[i : i + window_size]
            if len(window) >= self.min_topic_size:
                documents.append(" ".join(window))

        if not documents:
            logger.warning("No documents created from tokens")
            return {}

        # Optimize topic number if not specified
        if num_topics is None:
            num_topics = self.optimize_topic_number(documents)
            logger.info(f"Optimal number of topics: {num_topics}")

        # Create vocabulary and document-term matrix
        vectorizer = TfidfVectorizer(
            max_features=100, ngram_range=(1, 2), min_df=0.1, max_df=0.9
        )

        try:
            doc_term_matrix = vectorizer.fit_transform(documents)
            vocab = vectorizer.get_feature_names_out()
        except:
            logger.error("Failed to create document-term matrix")
            return {}

        # Build semantic similarity matrix
        similarity_matrix = self.build_semantic_similarity_matrix(list(vocab))

        # Create enhanced LDA model with semantic regularization
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            learning_method="online",
            max_iter=50,
        )

        # Fit the model
        lda.fit(doc_term_matrix)

        # Extract topics with semantic enhancement
        topics = []
        feature_names = vocab

        for topic_idx, topic in enumerate(lda.components_):
            # Get top words for this topic
            top_words_idx = topic.argsort()[-10:][::-1]

            # Enhance with semantically similar words
            enhanced_words = set()
            for idx in top_words_idx:
                enhanced_words.add(feature_names[idx])

                # Add semantically similar words
                similar_idx = np.where(similarity_matrix[idx] > 0.5)[0]
                for sim_idx in similar_idx[:3]:  # Add top 3 similar words
                    enhanced_words.add(feature_names[sim_idx])

            topics.append({
                "topic_id": topic_idx,
                "words": list(enhanced_words)[:10],  # Top 10 words
                "weights": topic[top_words_idx].tolist(),
            })

        self.topics = topics
        self.lda_model = lda

        return {
            "topics": topics,
            "num_topics": num_topics,
            "coherence_scores": self.coherence_scores,
            "model": lda,
        }

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
            hypernyms = synsets[0].hypernyms()
            if hypernyms:
                # Return the simplest hypernym
                return hypernyms[0].lemmas()[0].name().replace('_', ' ')
        return ''

    def extract_topics(self, num_topics: Optional[int] = None) -> List[Dict]:
        """
        Main method to extract topics from the blog content using the full
        transformer-enhanced LDA pipeline.

        Parameters:
        -----------
        num_topics : Optional[int]
            Number of topics to extract (if None, will be optimized)

        Returns:
        --------
        List[Dict] : Extracted topics with labels and relevant information
        """
        logger.info("Starting topic extraction pipeline...")

        # Run the full pipeline
        results = self.enhanced_lda_modeling(num_topics)

        if not results:
            logger.error("Topic modeling failed")
            return []

        # Refine topics
        refined_topics = self.refine_topic_labels()

        logger.info(f"Extracted {len(refined_topics)} topics successfully")

        return refined_topics

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
        
    def get_topic_summary(self) -> dict[str, Any]:
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

        print(summary)
        return summary


# Example usage and testing
if __name__ == "__main__":
    # Example blog content (replace with actual blog data)
    sample_blog = """
      Yesterday I went to the drive in and saw the movie "The Day After Tomorrow" It wasn't that great. It had some good parts, but not worth all the hype it got. Anywyas at the drive in there were so many mexicans there so of course I was smelly the "mary jane" all night long. Man how I wish I had some of that right now. I wish I could get into it. Today I was out caughting bugs for my stupid bio project. It's so gay. I'm getting like an F on it. I know I will. I only got like 16 bugs out of 30. And only 1 week left. Atleast I don't have skool tomorrow. Also on friday my counceler was questioning if I should goto summer skool or not. She better say yes or I will be so pissed.  I saw a great japanese show today even though I forget what it was mostly about. But I do now it was great. The korean show I watch is on tomorrow. Also the one I watch Wed. and Thurs. I think it is ending soon. That will so suck. But a new one will come on. I hope it is just as good.  Just like 1 hour ago I saw this think on nostradamus. It was so crazy. All them predictions were like so close. I wonder what the future holds...
    """

    # Initialize the model
    topic_model = TransformerEnhancedLDA(sample_blog, min_topic_size=8)

    # Extract topics
    topics = topic_model.extract_topics(num_topics=12)

    # Hierachy of topics
    htopics =  topic_model.evaluate_topics()
    
    print(htopics)
