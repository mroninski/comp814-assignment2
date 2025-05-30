import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import re

# Download required NLTK data (run once)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("punkt", quiet=True)


def extract_topics_lda(content, n_topics=5, preprocessing_pipeline="standard"):
    """
    Extract topics from text content using LDA with different preprocessing pipelines.

    Parameters:
    -----------
    content : str or list
        The text content to analyze (single string or list of strings)
    n_topics : int
        Number of topics to extract (default: 5)
    preprocessing_pipeline : str
        Pipeline type: 'standard', 'minimal', 'aggressive', 'pos_filtered'

    Returns:
    --------
    dict : Dictionary containing:
        - 'generic_topics': List of tuples (topic_words, topic_score) for broad topics
        - 'specific_topics': List of tuples (topic_words, topic_score) for specific topics
        - 'dominant_topic_idx': Index of the most dominant topic
        - 'topic_distribution': Probability distribution over topics
    """

    # Ensure content is a list for consistency
    if isinstance(content, str):
        documents = [content]
    else:
        documents = content

    # Statistical stop words (using existing NLTK corpus)
    stop_words = set(stopwords.words("english"))

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Preprocessing pipelines
    def preprocess_text(text, pipeline):
        # Clean non-ASCII characters
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        text = text.lower()

        if pipeline == "minimal":
            # Minimal preprocessing - only remove numbers and punctuation
            text = re.sub(r"[^a-zA-Z\s]", " ", text)
            tokens = text.split()

        elif pipeline == "standard":
            # Standard preprocessing with statistical stop words
            text = re.sub(r"[^a-zA-Z\s]", " ", text)
            tokens = word_tokenize(text)
            tokens = [
                lemmatizer.lemmatize(token)
                for token in tokens
                if token not in stop_words
            ]

        elif pipeline == "aggressive":
            # Aggressive preprocessing with length filtering
            text = re.sub(r"[^a-zA-Z\s]", " ", text)
            tokens = word_tokenize(text)
            tokens = [
                lemmatizer.lemmatize(token)
                for token in tokens
                if token not in stop_words and len(token) > 2
            ]

        elif pipeline == "pos_filtered":
            # POS-based filtering for nouns and proper nouns
            text = re.sub(r"[^a-zA-Z\s]", " ", text)
            tokens = word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            # Keep only nouns (NN*) and proper nouns (NNP*)
            tokens = [
                lemmatizer.lemmatize(word)
                for word, pos in pos_tags
                if pos.startswith("NN") and word not in stop_words
            ]

        return " ".join(tokens)

    # Apply preprocessing
    processed_docs = [preprocess_text(doc, preprocessing_pipeline) for doc in documents]

    # Filter out empty documents
    processed_docs = [doc for doc in processed_docs if doc.strip()]

    if not processed_docs:
        return {
            "generic_topics": [],
            "specific_topics": [],
            "dominant_topic_idx": -1,
            "topic_distribution": [],
        }

    # Vectorization parameters based on pipeline
    if preprocessing_pipeline in ["minimal", "standard"]:
        # Use both unigrams and bigrams for generic topics
        vectorizer_params = {
            "max_features": 100,
            "ngram_range": (1, 2),
            "max_df": 0.95,
            "min_df": 1,
        }
    else:
        # More restrictive for specific topics
        vectorizer_params = {
            "max_features": 50,
            "ngram_range": (1, 1),
            "max_df": 0.90,
            "min_df": 1,
        }

    # Create document-term matrix
    vectorizer = CountVectorizer(**vectorizer_params)
    doc_term_matrix = vectorizer.fit_transform(processed_docs)

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Apply LDA
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=50,
        learning_method="online",
        learning_offset=50.0,
        random_state=42,
        n_jobs=-1,
    )

    # Fit LDA model
    doc_topic_dist = lda.fit_transform(doc_term_matrix)

    # Extract topics with their weights
    def get_top_words(topic_weights, feature_names, n_words=5):
        top_indices = topic_weights.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_scores = [topic_weights[i] for i in top_indices]
        return list(zip(top_words, top_scores))

    # Get all topics
    all_topics = []
    for topic_idx, topic in enumerate(lda.components_):
        normalized_topic = topic / topic.sum()
        topic_words = get_top_words(normalized_topic, feature_names)
        all_topics.append((topic_words, doc_topic_dist[0][topic_idx]))

    # Sort topics by relevance to document
    all_topics.sort(key=lambda x: x[1], reverse=True)

    # Classify topics as generic or specific based on word scores distribution
    generic_topics = []
    specific_topics = []

    for topic_words, topic_score in all_topics:
        # Calculate entropy of word scores to determine specificity
        word_scores = [score for _, score in topic_words]
        word_scores_norm = np.array(word_scores) / np.sum(word_scores)
        entropy = -np.sum(word_scores_norm * np.log(word_scores_norm + 1e-10))

        # Higher entropy = more generic topic (words have similar weights)
        # Lower entropy = more specific topic (few words dominate)
        if entropy > np.log(len(topic_words)) * 0.7:  # Statistical threshold
            generic_topics.append((topic_words, topic_score))
        else:
            specific_topics.append((topic_words, topic_score))

    # Get dominant topic
    dominant_topic_idx = np.argmax(doc_topic_dist[0])

    return {
        "generic_topics": generic_topics[:2],  # Top 2 generic topics
        "specific_topics": specific_topics[:2],  # Top 2 specific topics
        "dominant_topic_idx": dominant_topic_idx,
        "topic_distribution": doc_topic_dist[0].tolist(),
    }


# Usage example for single document processing (for aggregation later)
# def process_single_blog(blog_content):
#     """
#     Process a single blog post and extract its topics.
#     Returns simplified format for aggregation.
#     """
#     results = extract_topics_lda(
#         blog_content, n_topics=3, preprocessing_pipeline="standard"
#     )

#     # Simplified output for aggregation
#     topics = []

#     # Add generic topics
#     for topic_words, score in results["generic_topics"]:
#         topic_string = " ".join([word for word, _ in topic_words[:3]])  # Top 3 words
#         topics.append(("generic", topic_string, score))

#     # Add specific topics
#     for topic_words, score in results["specific_topics"]:
#         topic_string = " ".join([word for word, _ in topic_words[:3]])  # Top 3 words
#         topics.append(("specific", topic_string, score))

#     return topics
