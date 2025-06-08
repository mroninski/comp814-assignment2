"""
Text Mining Transformer
==============================================

This module provides a comprehensive class for performing text mining transformations
on Polars DataFrames, specifically designed for the COMP814 Text Mining Assignment.

The intent is to process all of the transformations to the content in this one file.

"""

import json
import logging
import re
from collections import Counter
from typing import List
import os

import nltk
import polars as pl
from deep_translator import GoogleTranslator
from langdetect import LangDetectException, detect


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Download required NLTK data
try:
    nltk.data.find("punkt")
    nltk.data.find("stopwords")
    nltk.data.find("wordnet")
except IndexError as ie:
    logging.warning(f"IndexError: {ie}")
    pass
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

# This is to avoid the tokenizers parallelism error
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PostsTableTransformation:
    """
    A class for performing text mining transformations on Polars DataFrames.

    This class implements language detection, translation, and word frequency
    analysis following text mining best practices as outlined in Manning & Schütze (1999)
    and Jurafsky & Martin (2023).

    Attributes:
        df (pl.DataFrame): The input Polars DataFrame
        stop_words (set): Set of English stopwords
        lemmatizer (WordNetLemmatizer): NLTK lemmatizer for word normalization
        logger (logging.Logger): Logger for tracking transformations
    """

    def __init__(self, df: pl.LazyFrame, remove_stopwords: bool = True):
        """
        Initialize the PostsTableTransformation.

        Args:
            df (pl.DataFrame): Input DataFrame with schema containing 'content' column
            remove_stopwords (bool): Whether to remove stopwords in word frequency analysis

        Raises:
            ValueError: If the DataFrame doesn't contain required columns
        """
        self._validate_dataframe(df)
        self.df = df
        self.remove_stopwords = remove_stopwords

        # Initialize NLP tools
        self.stop_words = set(stopwords.words("english")) if remove_stopwords else set()
        self.lemmatizer = WordNetLemmatizer()

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _validate_dataframe(self, df: pl.LazyFrame) -> None:
        """Validate that the DataFrame contains required columns."""
        required_columns = ["content"]
        schema = df.schema

        for col in required_columns:
            if col not in schema:
                raise ValueError(f"DataFrame must contain '{col}' column")

    def detect_language(self) -> "PostsTableTransformation":
        """
        Detect the language of each content entry.

        This method uses the langdetect library, which implements a language detection
        algorithm based on Naive Bayes classifier (Shuyo, 2010).

        Returns:
            self: Returns self for method chaining

        Note:
            - Short texts may result in less accurate detection
            - Empty or non-text content will be marked as 'unknown'
        """
        self.logger.info("Starting language detection...")

        def _detect_language_safe(text: str) -> str:
            """Safely detect language with error handling."""
            if not text or not isinstance(text, str) or len(text.strip()) < 3:
                return "unknown"

            try:
                # Clean text for better detection
                clean_text = re.sub(r"[^\w\s]", " ", text)
                clean_text = re.sub(r"\s+", " ", clean_text).strip()

                if len(clean_text) < 10:  # Too short for reliable detection
                    return "unknown"

                return detect(clean_text)
            except LangDetectException:
                return "unknown"
            except Exception as e:
                self.logger.warning(f"Error detecting language: {e}")
                return "unknown"

        # Apply language detection
        self.df = self.df.with_columns(
            pl.col("content")
            .map_elements(_detect_language_safe, return_dtype=pl.Utf8)
            .alias("content_language")
        )

        # Log language distribution
        lang_counts = self.df.group_by("content_language").count()
        self.logger.info(f"Language distribution:\n{lang_counts}")

        return self

    def translate_to_english(self, batch_size: int = 50) -> "PostsTableTransformation":
        """
        Translate non-English content to English.

        This method uses the deep-translator library with Google Translate API.
        Following best practices from Koehn (2020), we preserve original text
        for English content and handle translation errors gracefully.

        Args:
            batch_size (int): Number of texts to translate in each batch

        Returns:
            self: Returns self for method chaining

        Note:
            - Translation quality may vary depending on source language
            - Network connection required for translation
            - Rate limiting may apply for large datasets
        """
        self.logger.info("Starting translation to English...")

        # Check if language detection has been performed
        if "content_language" not in self.df.columns:
            self.logger.warning(
                "Language detection not performed. Running detection first..."
            )
            self.detect_language()

        def _translate_text(text: str, lang: str) -> str:
            """Translate text to English if not already English."""
            if not text or lang in ["en", "unknown"] or not isinstance(text, str):
                return text

            try:
                # Limit text length to avoid API limits
                if len(text) > 5000:
                    text = text[:5000]

                translator = GoogleTranslator(source=lang, target="en")
                translated = translator.translate(text)

                return translated if translated else text
            except Exception as e:
                self.logger.warning(f"Translation error for language '{lang}': {e}")
                return text

        # Create a temporary DataFrame with row numbers for batch processing
        df_with_row_nr = self.df.with_row_count()

        # Process in batches to avoid API rate limits
        translated_texts = []
        total_rows = df_with_row_nr.select(pl.len()).collect()

        for i in range(0, total_rows, batch_size):
            batch_end = min(i + batch_size, total_rows)
            batch_df = df_with_row_nr.filter(
                (pl.col("row_nr") >= i) & (pl.col("row_nr") < batch_end)
            )

            # Translate batch
            batch_translated = []
            for row in batch_df.iter_rows(named=True):
                if row["content_language"] == "en":
                    batch_translated.append(row["content"])
                    continue
                translated = _translate_text(
                    row["content"], row.get("content_language", "unknown")
                )
                batch_translated.append(translated)

            translated_texts.extend(batch_translated)

            if (i + batch_size) % 500 == 0:
                self.logger.info(
                    f"Translated {min(i + batch_size, total_rows)}/{total_rows} texts"
                )

        # Add translated column
        self.df = self.df.with_columns(pl.Series("content_english", translated_texts))

        self.logger.info("Translation completed")
        return self

    def clean_up_content_column(self) -> "PostsTableTransformation":
        """
        Clean up the content column by removing special characters and annoying values
        """
        # Remove any non-ascii characters from the content column
        # Also remove the value `urlLink` from the content column
        # Remove the value `nbsp` from the content column and anything similar from a HTML entity
        self.df = self.df.with_columns(
            pl.col("content")
            .str.replace_all(r"[^a-zA-Z\s]", " ")
            .str.replace_all("urlLink", "")
            .str.replace_all("nbsp", "")
            .str.replace_all("&amp;", "&")
            .str.replace_all("&lt;", "<")
            .str.replace_all("&gt;", ">")
            .str.replace_all("&quot;", '"')
            .str.replace_all("&#39;", "'")
            .str.replace_all("&ndash;", "-")
            .str.replace_all("&mdash;", "-")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )

        return self

    def clean_up_industry_column(self) -> "PostsTableTransformation":
        """
        Clean up the industry column by replacing `indUnknown` with `unknown`
        """
        self.df = self.df.with_columns(
            pl.col("industry").str.replace_all("indUnknown", "unknown")
        )
        return self

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for word frequency analysis.

        Following Jurafsky & Martin (2023) recommendations:
        1. Lowercase conversion
        2. Remove special characters and digits
        3. Tokenization
        4. Stopword removal (optional)
        5. Lemmatization

        Args:
            text (str): Input text

        Returns:
            List[str]: List of preprocessed tokens
        """
        if not text or not isinstance(text, str):
            return []

        # Lowercase and remove special characters
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords if enabled
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]

        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        # Filter out empty tokens and single characters
        tokens = [token for token in tokens if len(token) > 1]

        return tokens

    def compute_word_frequencies(
        self,
        n_most_common: int = 10,
        n_least_common: int = 10,
        min_word_length: int = 2,
    ) -> "PostsTableTransformation":
        """
        Compute most and least common words in the content.

        This method implements TF (Term Frequency) analysis as described in
        Manning et al. (2008). For more sophisticated analysis, consider
        using TF-IDF weighting.

        Args:
            n_most_common (int): Number of most common words to extract
            n_least_common (int): Number of least common words to extract
            min_word_length (int): Minimum word length to consider

        Returns:
            self: Returns self for method chaining

        Note:
            - Uses 'content_english' column if available, otherwise 'content'
            - Words appearing only once are excluded from least common to avoid noise
        """
        self.logger.info("Computing word frequencies...")

        # Determine which column to use
        content_column = (
            "content_english" if "content_english" in self.df.columns else "content"
        )

        def _compute_frequencies(text: str) -> dict:
            """Compute most and least common words for a single text."""
            tokens = self._preprocess_text(text)

            if not tokens:
                return {"most_common": [], "least_common": []}

            # Count word frequencies
            word_freq = Counter(tokens)

            # Filter by minimum word length
            word_freq = {
                word: count
                for word, count in word_freq.items()
                if len(word) >= min_word_length
            }

            if not word_freq:
                return {"most_common": [], "least_common": []}

            # Get most common words
            most_common = Counter(word_freq).most_common(n_most_common)
            most_common_words = [word for word, _ in most_common]

            # Get least common words (excluding hapax legomena)
            # Filter words that appear more than once to reduce noise
            filtered_freq = {
                word: count for word, count in word_freq.items() if count > 1
            }

            if filtered_freq:
                least_common = sorted(filtered_freq.items(), key=lambda x: x[1])[
                    :n_least_common
                ]
                least_common_words = [word for word, _ in least_common]
            else:
                least_common_words = []

            return {
                "most_common": most_common_words,
                "least_common": least_common_words,
            }

        # Add columns to DataFrame
        self.df = self.df.with_columns(
            pl.col(content_column)
            .map_elements(
                lambda x: json.dumps(_compute_frequencies(x), default=str),
                return_dtype=pl.Utf8,
            )
            .alias("word_frequencies")
        )

        self.logger.info("Word frequency computation completed")
        return self

    def get_dataframe(self) -> pl.LazyFrame:
        """
        Get the transformed DataFrame.

        Returns:
            pl.DataFrame: The DataFrame with all applied transformations
        """
        return self.df


# Example usage
if __name__ == "__main__":
    df = pl.LazyFrame(pl.read_parquet(".data/tables/posts_df.parquet"))

    print("Original DataFrame:")
    print(df)
    print("\n" + "=" * 50 + "\n")

    # Initialize transformer
    transformer = PostsTableTransformation(df, remove_stopwords=True)

    # Apply transformations using method chaining
    transformed_df = (
        transformer.detect_language()
        .compute_word_frequencies(n_most_common=5, n_least_common=3)
        .get_dataframe()
    )

    # Keep only a random sample of 100 rows
    transformed_df = transformed_df.limit(5)

    print("Transformed DataFrame:")
    print(transformed_df)
    print("\n" + "=" * 50 + "\n")

    # Display specific columns for clarity
    print("Language Detection Results:")
    print(transformed_df.select(["content", "content_language"]).collect())
    print("\n" + "=" * 50 + "\n")
    print("\n" + "=" * 50 + "\n")

    print("Word Frequency Results:")
    print(transformed_df.select(["file_id", "word_frequencies"]))
    print("\n" + "=" * 50 + "\n")
