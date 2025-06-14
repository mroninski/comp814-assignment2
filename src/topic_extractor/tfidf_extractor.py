import spacy
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from sklearn.feature_extraction.text import TfidfVectorizer
import polars as pl
import json
import numpy as np
import re

from typing import Dict, List, Optional, Union


class TFIDFTopicExtractor:
    def __init__(self, top_n: int = 5, min_topic_size: int = 5, max_features: int = 1000):
        self.top_n = top_n
        self.min_topic_size = min_topic_size
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words='english')
        self.nlp_model = spacy.load("en_core_web_sm")

    def _clean_text(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Clean raw blog text or list of texts by removing HTML artifacts and noise.

        Args:
            text: Raw blog content as a string or list of strings

        Returns:
            Cleaned text or list of cleaned texts suitable for topic modeling
        """

        def clean_single(t: str) -> str:
            # remove HTML tags
            t = re.sub(r"<[^>]+>", " ", t)
            # decode common HTML entities
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
                t = t.replace(entity, replacement)

            # remove URLs and emails
            t = re.sub(r"http[s]?://\S+", " ", t)
            t = re.sub(r"\S+@\S+", " ", t)

            # normalize whitespace
            t = re.sub(r"\s+", " ", t).strip()
            return t

        if isinstance(text, list):
            return [clean_single(t) for t in text if isinstance(t, str)]
        else:
            return clean_single(text)

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

    def _create_semantic_documents(self, content: Union[str, List[str]]) -> List[str]:
        """
        Create semantic documents using sentence boundaries for better topic coherence.
        Accepts either a single string or a list of blog content strings.

        Args:
            content: Cleaned blog content (string or list of strings)

        Returns:
            List of semantic documents for topic modeling
        """
        if isinstance(content, str):
            content = [content]

        documents = []
        for entry in content:
            if not isinstance(entry, str):
                continue

            doc = self.nlp_model(entry)
            sentences = [
                sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20
            ]

            # Group sentences into coherent documents (2-3 sentences each)
            for i in range(0, len(sentences), 2):
                doc_text = " ".join(sentences[i: i + 3])
                if len(doc_text.split()) >= self.min_topic_size:
                    documents.append(doc_text)

            # Fallback to token-based windowing if insufficient documents
            if len(documents) < 3:
                tokens = self._preprocess_text(entry)
                window_size, overlap = 30, 10
                for i in range(0, len(tokens), window_size - overlap):
                    window = tokens[i: i + window_size]
                    if len(window) >= self.min_topic_size:
                        documents.append(" ".join(window))

        return documents

    def extract_topics(self, blog_content: str, num_words: int = 10) -> Dict:
        clean_content = self._clean_text(blog_content)
        tokens = self._preprocess_text(clean_content)
        document = " ".join(tokens)

        if not document.strip():
            return {
                "tfidf_results": {"topics": [], "num_topics": 0, "document_count": 0, "average_topic_quality": 0.0},
                "words": set(),
                "topic_labels": []
            }

        # Vectorize single document (wrapped in list)
        tfidf_matrix = self.vectorizer.fit_transform([document])
        feature_names = self.vectorizer.get_feature_names_out()
        row = tfidf_matrix.toarray().flatten()

        # Get top-N indices
        top_indices = np.argsort(row)[::-1][:self.top_n]
        top_words = [feature_names[i] for i in top_indices if row[i] > 0]
        top_weights = [row[i] for i in top_indices if row[i] > 0]

        return {
            "tfidf_results": {
                "topics": [{
                    "topic_id": 0,
                    "label": " ".join(top_words[:2]),  # or just use a static label like "blog_topic"
                    "words": top_words[:num_words],
                    "weights": top_weights[:num_words],
                    "topic_quality": float(np.mean(top_weights[:num_words]))
                }] if top_words else [],
                "num_topics": 1 if top_words else 0,
                "document_count": 1,
                "average_topic_quality": float(np.mean(top_weights)) if top_weights else 0.0
            },
            "words": set(top_words),
            "topic_labels": top_words[:2]
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

