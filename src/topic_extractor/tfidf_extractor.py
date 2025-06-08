import spacy
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from sklearn.feature_extraction.text import TfidfVectorizer
import polars as pl
import json
import numpy as np
import re

from typing import Dict, List, Optional


class TFIDFTopicExtractor:
    def __init__(self, top_n: int = 5, min_topic_size: int = 5, max_features: int = 1000):
        self.top_n = top_n
        self.min_topic_size = min_topic_size
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words='english')
        self.nlp_model = spacy.load("en_core_web_sm")


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

    # def __init__(self, min_topic_size: int = 5):
    #     """
    #     Initialize the topic extraction model.
    #
    #     Args:
    #         min_topic_size: Minimum number of words required to form a meaningful topic
    #     """
    #     self.min_topic_size = min_topic_size
    #     self._initialize_models()

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
            doc_text = " ".join(sentences[i: i + 3])
            if len(doc_text.split()) >= self.min_topic_size:
                documents.append(doc_text)

        # Fallback to token-based windowing if insufficient documents
        if len(documents) < 3:
            tokens = self._preprocess_text(content)
            window_size, overlap = 30, 10

            for i in range(0, len(tokens), window_size - overlap):
                window = tokens[i: i + window_size]
                if len(window) >= self.min_topic_size:
                    documents.append(" ".join(window))

        return documents

    def extract_topics(self, blog_content: str, num_words: int = 10) -> Dict:
        # Preprocess text
        clean_content = self._clean_text(blog_content)
        documents = self._create_semantic_documents(clean_content)

        if not documents:
            return {
                "tfidf_results": {"topics": [], "num_topics": 0, "document_count": 0, "average_topic_quality": 0.0},
                "words": set(),
                "topic_labels": []
            }

        # Create TF-IDF matrix
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        feature_names = self.vectorizer.get_feature_names_out()

        topics = []
        all_words = set()
        topic_labels = []

        # Process each document row
        for doc_idx, row in enumerate(doc_term_matrix):
            row_array = row.toarray().flatten()
            top_indices = np.argsort(row_array)[::-1][:self.top_n]

            top_words = [feature_names[i] for i in top_indices if row_array[i] > 0]
            top_weights = [row_array[i] for i in top_indices if row_array[i] > 0]

            if top_words:
                topic_label = self._generate_topic_label(top_words)
                all_words.update(top_words)
                topic_labels.extend(topic_label.split("_"))

                topics.append({
                    "topic_id": doc_idx,
                    "label": topic_label,
                    "words": top_words[:num_words],
                    "weights": top_weights[:num_words],
                    "topic_quality": float(np.mean(top_weights[:num_words]))
                })

        meaningful_topics = self._filter_meaningful_tfidf_topics(topics)

        return {
            "tfidf_results": {
                "topics": meaningful_topics,
                "num_topics": len(meaningful_topics),
                "document_count": len(documents),
                "average_topic_quality": float(
                    np.mean([t["topic_quality"] for t in meaningful_topics])
                ) if meaningful_topics else 0.0
            },
            "words": all_words,
            "topic_labels": topic_labels
        }

    def _filter_meaningful_tfidf_topics(self, topics: List[Dict]) -> List[Dict]:
        if not topics:
            return []

        qualities = [t["topic_quality"] for t in topics]
        max_quality = max(qualities)
        mean_quality = np.mean(qualities)

        threshold = 0.7 * max_quality if max_quality > 0.3 else max(0.8 * max_quality, mean_quality)
        threshold = max(threshold, 0.01)

        meaningful_topics = []
        for topic in sorted(topics, key=lambda x: x["topic_quality"], reverse=True):
            if topic["topic_quality"] >= threshold:
                topic_words = topic.get("words", [])
                if not isinstance(topic_words, list):
                    topic_words = list(topic_words)  # fallback for safety

                topic_set = set(topic_words[:5])
                is_unique = True

                for existing in meaningful_topics:
                    existing_words = existing.get("words", [])
                    if not isinstance(existing_words, list):
                        existing_words = list(existing_words)
                    existing_set = set(existing_words[:5])
                    overlap_ratio = len(topic_set & existing_set) / max(1, min(len(topic_set), len(existing_set)))
                    if overlap_ratio > 0.6:
                        is_unique = False
                        break

                if is_unique:
                    meaningful_topics.append(topic)

        return meaningful_topics[:3]


# Example usage for testing
if __name__ == "__main__":
    import json

    sample_blog = """
    Machine learning and artificial intelligence are transforming how we process data.
    Natural language processing enables computers to understand human language.
    Deep learning algorithms use neural networks to identify patterns in large datasets.
    These technologies are being applied in healthcare, finance, and autonomous vehicles.
    """

    # Initialize your TF-IDF topic extractor
    topic_model = TFIDFTopicExtractor(max_features=1000, top_n=5)

    # Call extract_topics with a list of documents (even if just one)
    results = topic_model.extract_topics(sample_blog)

    print("TF-IDF TOPIC EXTRACTION RESULTS")
    print("=" * 50)

    for i, topic in enumerate(results["tfidf_results"]["topics"]):
        print(f"Topic {i + 1} | Label: {topic['label']}")
        print(f"Top Words: {topic['words']}")
        print(f"Quality: {topic['topic_quality']:.3f}")
        print("-" * 30)

