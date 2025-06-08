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
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.cli.download import download

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
        documents = self._create_semantic_documents(blog_content)

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

        # Step 5: Create document-term matrix.
        # For LDA, a richer vocabulary is often better. We use max_features=1000
        # to provide the model with more words to build topics from, which is crucial for
        # capturing nuanced themes in short texts. min_df=2 helps filter out noise by
        # ignoring terms that appear in only one semantic document chunk.
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.85,
            stop_words="english",
            lowercase=True,
            sublinear_tf=True,
            smooth_idf=True,
        )

        try:
            doc_term_matrix = vectorizer.fit_transform(enhanced_documents)
            feature_names = vectorizer.get_feature_names_out()
        except ValueError:
            # Fallback for very short content where min_df might be too strict
            vectorizer_fallback = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.85,
                stop_words="english",
                lowercase=True,
                sublinear_tf=True,
                smooth_idf=True,
            )
            doc_term_matrix = vectorizer_fallback.fit_transform(enhanced_documents)
            feature_names = vectorizer_fallback.get_feature_names_out()
        except Exception as e:
            logger.error(f"Failed to create document-term matrix: {e}")
            return {"lda_results": {"topics": []}, "words": set(), "topic_labels": []}

        # Step 6: Train LDA model with optimized parameters.
        # We switch to 'online' learning, which is significantly faster for larger datasets
        # and well-suited for a pipeline processing many records. It uses mini-batch updates,
        # avoiding costly passes over the entire dataset. Consequently, max_iter can be reduced.
        # The priors (doc_topic_prior, topic_word_prior) are set to low values, a common
        # practice for short texts to encourage sparser topic and word distributions.
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            learning_method="online",
            max_iter=20,
            doc_topic_prior=0.1,
            topic_word_prior=0.01,
            learning_offset=50.0,
        )
        lda.fit(doc_term_matrix)

        # Step 7: Extract, evaluate, and semantically re-rank topics.
        # The following section processes the raw LDA output. It enriches the topics by
        # re-ranking words based on their semantic similarity to a topic's embedding centroid,
        # rather than relying solely on the probabilistic weights from LDA. This enhances
        # topic interpretability.
        vocab_embeddings = self.sentence_transformer.encode(
            list(feature_names), convert_to_numpy=True
        )
        topics = []

        for topic_idx, topic_weights in enumerate(lda.components_):
            # 1. Get initial top words based on LDA's learned topic-word distribution.
            top_indices = topic_weights.argsort()[-20:][::-1]
            top_words = [str(feature_names[idx]) for idx in top_indices]
            top_weights = topic_weights[top_indices].tolist()

            # 2. Semantically refine the topic words.
            # This moves beyond LDA's bag-of-words assumption by using transformer embeddings.
            if len(top_indices) > 0:
                # Create a semantic "center" for the topic using the embeddings of its top words.
                topic_centroid = np.mean(vocab_embeddings[top_indices[:8]], axis=0)
                coherence_scores = []
                coherent_words, coherent_weights = [], []

                # 3. Re-rank words based on their cosine similarity to the topic centroid.
                # This ensures the final words are not just probable but also semantically related.
                for i, word_idx in enumerate(top_indices):
                    word_embedding = vocab_embeddings[word_idx]
                    similarity = 1 - cosine(topic_centroid, word_embedding)

                    # Words are kept if they are semantically close to the centroid.
                    if (similarity > 0.25 and i < 15) or similarity > 0.35:
                        coherent_words.append(top_words[i])
                        coherent_weights.append(top_weights[i])
                        coherence_scores.append(similarity)

                # Fallback to ensure a minimum number of words per topic.
                if len(coherent_words) < 5:
                    coherent_words = top_words[:8]
                    coherent_weights = top_weights[:8]
                    coherence_scores = [0.5] * len(coherent_words)
            else:
                coherent_words, coherent_weights, coherence_scores = [], [], []

            # 4. Calculate a composite quality score for the refined topic.
            # This score is used later to filter out low-quality or nonsensical topics.
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

        # Step 8: Filter meaningful topics and prepare output
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
    import time

    sample_blog = """Well this is my first entry and id like to start by saying I ve only started this because my computing class is so fucking boaring I go to Waihi college and its a decent school with alot of fucking weirdos and mentaly divergent people who think they know every thing and are loved by every one this town called Waihi is a hole its a small town thats soully dependant on a mine thats closing in about years this place is a HOLE evry one leaves as soon as they can and if they dont they end up staying for ever I  live with my mates in this renivated garage thast got its own bathroom toilet facilitys and the works I ve got a year old girl friend who s the shit and oh so beautiful I ve got heaps of mates of who im not going to name in this  except one who has his own blog here his names Mattew will s or Murk its friday im board i cant wait to get home to get absolutley Motherd and play a little X box then go out or some thing maybe get some subway or  KFC  if you read this i hope you got some injoyment from reading about some one elses life and what they get up to  If you read this I ask you Why you should be out side or in town at a mates or fucking out clubing not sitting in some pour conditioned chair at home with most the curtains closed reading about other peoples lives to make your self feel alive F I N  Sorry about the spelling Its monday and im in computing again its as boaring as always and we have a relever this time I geuss if you have read my blog you can tell im quite the hipocrate of sorts What i mean is I was a dick about the reading my blog thing at the end but its true to some extent If you have time to read this you have time to do other things as well  Im going for my lisence in a few days I know i should have it by now but i just havent gotten round to it I should be able to do it my two mates are worrying about there lives after school and what unis to go to and if thay should split up and go to different places or not They should just go with it For me who knows what ill do may be ill go to wellinton with my other mates and do a computing course or go to auckland with some girl mates of mine work with my uncle in auckland also or go to hamilton and flat with my mate even go down to otago and stay down there if i wanted with one of my mates But i more then lickly will go to australia to work on a tourisum based job living with my aunti since shes offered It will be a change for a while Who knows aye from other blogs I see you guys who read these you don t really what to know a bout us but about what we do and the people we know Any way I got a hair cut last night it was weird the stupid hair dressers have put there prices up fucking dollars for a cut instead of My girl friend has brought a car last night it was a  Mazda its pretty good but not the best looking car my friend Matthew and his enemy Sam mudgway are still at it waging war against each other its more like guerilla warfare one of then will strike a blow and then the other will and my mate I live with he sat ends he needs to decide what he s doing for the rest of his life since hes going to uni but that s really indecisive at the moment with school and work and with the ordeal of getting over his ex even though hes gone out with this hot blonde he brock up with her because he keeps thinking of his ex he hasn t gone out with his ex for the last to months he needs to get over it really my other mate I live with brought a amp for his bass guitar and hes buying a guitar this weekend in tauranga Im going with just for the ride or I could spend the day with my girl friend but who would want to do that Just to get this off my chest I shouldn t have done what I did at the after ball I really shouldn t have well that s all for now"""

    print("PERFORMANCE TESTING - TOPIC EXTRACTION")
    print("=" * 60)
    print(f"Sample blog length: {len(sample_blog)} characters")
    print(f"Sample blog word count: {len(sample_blog.split())} words")

    # Test current implementation
    topic_model = TransformerEnhancedLDA(min_topic_size=2)

    # Run multiple iterations for more reliable timing
    num_runs = 3
    times = []

    for i in range(num_runs):
        print(f"\nRun {i + 1}/{num_runs}:")
        start_time = time.time()
        results = topic_model.extract_topics(sample_blog, num_topics=3)
        end_time = time.time()

        execution_time = end_time - start_time
        times.append(execution_time)

        print(f"  Execution time: {execution_time:.3f} seconds")
        print(
            f"  Topics extracted: {len(results.get('lda_results', {}).get('topics', []))}"
        )
        print(f"  Total words: {len(results.get('words', set()))}")

    avg_time = sum(times) / len(times)
    print(f"\nAVERAGE EXECUTION TIME: {avg_time:.3f} seconds")

    # Show detailed results from last run
    print("\nDETAILED RESULTS (Last Run):")
    print("=" * 40)
    for i, topic in enumerate(results.get("lda_results", {}).get("topics", [])):
        print(f"Topic {i + 1}: {topic.get('label', 'Unknown')}")
        print(f"  Quality: {topic.get('topic_quality', 0):.3f}")
        print(f"  Words: {', '.join(topic.get('words', [])[:5])}")
        print()
