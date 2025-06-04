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
    def __init__(self, max_features=1000, top_n=5):
        self.nlp_model = spacy.load("en_core_web_sm")
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.top_n = top_n


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

    def extract_topics(self, docs):
        # Step 1: Clean and preprocess
        cleaned_docs = [self._clean_text(doc) for doc in docs]
        enhanced_docs = self._enhance_vocabulary_with_phrases(cleaned_docs)

        # Step 2: TF-IDF vectorization
        tfidf_matrix = self.vectorizer.fit_transform(enhanced_docs)
        feature_names = self.vectorizer.get_feature_names_out()

        # Step 3: Extract top terms
        topics = []
        for row in tfidf_matrix:
            row_array = row.toarray().flatten()
            top_indices = np.argsort(row_array)[::-1][:self.top_n]
            top_terms = [feature_names[i] for i in top_indices if row_array[i] > 0]
            topics.append(json.dumps(top_terms))

        return pl.Series(topics)


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
    results = topic_model.extract_topics([sample_blog])

    print("TF-IDF TOPIC EXTRACTION RESULTS")
    print("=" * 50)
    for i, topic_json in enumerate(results):
        print(f"Doc {i+1} Top Terms: {json.loads(topic_json)}")

