#!/usr/bin/env python3
"""
Test file for improved LDA taxonomy mapping with proper weighting and scoring.
"""

import json
import sys
import os
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Add src to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from topic_extractor.topic_simplifying import (
    TopicTaxonomyMapper,
    map_lda_results_to_taxonomy,
)


def load_example_data() -> Dict[str, Any]:
    """Load example.json data for testing."""
    with open("example.json", "r") as f:
        return json.load(f)


def analyze_topic_words_and_weights(lda_data: Dict[str, Any]) -> None:
    """Analyze the structure of topic words and weights for debugging."""
    print("=== TOPIC ANALYSIS ===")

    topics = lda_data["lda_results"]["high_quality_topics"]

    for topic in topics:
        print(f"\nTopic {topic['topic_id']}: {topic['label']}")
        print(f"Topic Quality: {topic['topic_quality']:.4f}")
        print(f"Semantic Coherence: {topic['semantic_coherence']:.4f}")

        words = topic["words"]
        weights = topic["weights"]
        coherence_scores = topic["coherence_scores"]

        print("Top weighted words:")
        word_weight_pairs = list(zip(words, weights, coherence_scores))
        word_weight_pairs.sort(key=lambda x: x[1], reverse=True)

        for i, (word, weight, coherence) in enumerate(word_weight_pairs[:8]):
            print(
                f"  {i + 1}. '{word}': weight={weight:.4f}, coherence={coherence:.4f}"
            )


def test_current_implementation():
    """Test the current implementation to see what it produces."""
    print("\n=== TESTING CURRENT IMPLEMENTATION ===")

    # Load data
    lda_data = load_example_data()

    # Initialize mapper
    mapper = TopicTaxonomyMapper()

    # Test current implementation
    results = map_lda_results_to_taxonomy(mapper, lda_data, top_n=10)

    print("Current results:")
    for key, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {key}: {score}")

    return results


def test_weighted_word_selection(
    lda_data: Dict[str, Any], weight_percentile: float = 70
):
    """Test improved word selection using weight percentiles instead of fixed thresholds."""
    print(
        f"\n=== TESTING WEIGHTED WORD SELECTION (top {100 - weight_percentile}% by weight) ==="
    )

    topics = lda_data["lda_results"]["high_quality_topics"]

    for topic in topics:
        words = topic["words"]
        weights = topic["weights"]

        # Use percentile-based selection instead of fixed threshold
        weight_threshold = np.percentile(weights, weight_percentile)

        selected_words = []
        selected_weights = []
        for word, weight in zip(words, weights):
            if weight >= weight_threshold:
                selected_words.append(word)
                selected_weights.append(weight)

        print(f"\nTopic {topic['topic_id']}: {topic['label']}")
        print(f"Weight threshold (p{weight_percentile}): {weight_threshold:.4f}")
        print(f"Selected {len(selected_words)}/{len(words)} words:")

        # Sort by weight for display
        word_weight_pairs = list(zip(selected_words, selected_weights))
        word_weight_pairs.sort(key=lambda x: x[1], reverse=True)

        for word, weight in word_weight_pairs:
            print(f"  '{word}': {weight:.4f}")


def improved_weighted_embedding(
    words: List[str], weights: List[float], coherence_scores: List[float], mapper
) -> Optional[np.ndarray]:
    """
    Create improved weighted embedding that considers both weights and coherence.
    """
    embeddings = []
    final_weights = []

    for word, weight, coherence in zip(words, weights, coherence_scores):
        embedding = mapper._get_phrase_embedding(word)
        if embedding is not None:
            # Combine weight and coherence for final importance
            # Higher weight and higher coherence both contribute positively
            combined_weight = weight * (
                coherence + 0.1
            )  # Add 0.1 to avoid zero multiplication
            embeddings.append(embedding)
            final_weights.append(combined_weight)

    if not embeddings:
        return None

    # Normalize weights
    final_weights = np.array(final_weights)
    final_weights = final_weights / np.sum(final_weights)

    # Weighted average instead of simple average
    weighted_embedding = np.average(embeddings, axis=0, weights=final_weights)

    return weighted_embedding


def test_improved_mapping():
    """Test improved mapping with better weighting and quality scoring."""
    print("\n=== TESTING IMPROVED MAPPING ===")

    # Load data
    lda_data = load_example_data()

    # Initialize mapper
    mapper = TopicTaxonomyMapper()

    topics = lda_data["lda_results"]["high_quality_topics"]
    all_results = {}

    for topic in topics:
        topic_id = topic["topic_id"]
        words = topic["words"]
        weights = topic["weights"]
        coherence_scores = topic["coherence_scores"]
        topic_quality = topic["topic_quality"]
        semantic_coherence = topic["semantic_coherence"]

        print(f"\nProcessing Topic {topic_id}: {topic['label']}")

        # Use top words by weight (top 60% or minimum 3 words)
        weight_threshold = max(np.percentile(weights, 60), 0.3)
        min_words = min(3, len(words))

        # Get word-weight-coherence tuples and sort by weight
        word_data = list(zip(words, weights, coherence_scores))
        word_data.sort(key=lambda x: x[1], reverse=True)

        # Select words either above threshold or top N
        selected_data = []
        for word, weight, coherence in word_data:
            if weight >= weight_threshold or len(selected_data) < min_words:
                selected_data.append((word, weight, coherence))

        selected_words = [item[0] for item in selected_data]
        selected_weights = [item[1] for item in selected_data]
        selected_coherence = [item[2] for item in selected_data]

        print(f"Selected words: {selected_words}")

        # Create improved weighted embedding
        topic_embedding = improved_weighted_embedding(
            selected_words, selected_weights, selected_coherence, mapper
        )

        if topic_embedding is None:
            print(f"No valid embeddings for topic {topic_id}")
            continue

        # Compute similarities with subtopics
        subtopic_similarities = mapper._compute_semantic_similarity(
            topic_embedding, mapper.subtopic_embeddings
        )

        # Apply topic quality weighting to similarities
        # Higher quality topics get higher confidence in their mappings
        quality_multiplier = topic_quality  # Use topic quality directly as multiplier

        weighted_similarities = {
            subtopic: similarity * quality_multiplier
            for subtopic, similarity in subtopic_similarities.items()
        }

        # Filter and format results
        min_similarity = 0.15  # Lower threshold since we're using quality weighting
        filtered_similarities = {
            subtopic: score
            for subtopic, score in weighted_similarities.items()
            if score >= min_similarity
        }

        # Convert to major:subtopic format
        topic_results = {}
        for subtopic, similarity in filtered_similarities.items():
            major_topic = mapper.subtopic_to_major[subtopic]
            key = f"{major_topic}:{subtopic}"
            percentage = round(similarity * 100, 2)
            topic_results[key] = percentage

        # Add to overall results
        all_results.update(topic_results)

        # Show top results for this topic
        sorted_topic_results = sorted(
            topic_results.items(), key=lambda x: x[1], reverse=True
        )
        print(f"Top mappings for topic {topic_id}:")
        for key, score in sorted_topic_results[:5]:
            print(f"  {key}: {score}")

    # Return all results above 25% threshold, sorted by score
    final_results = {k: v for k, v in all_results.items() if v >= 25.0}
    final_results = dict(
        sorted(final_results.items(), key=lambda x: x[1], reverse=True)
    )

    print(f"\n=== FINAL IMPROVED RESULTS (>= 25%) ===")
    for key, score in final_results.items():
        print(f"{key}: {score}")

    return final_results


def compare_results():
    """Compare current vs improved implementation."""
    print("\n" + "=" * 60)
    print("COMPARISON OF IMPLEMENTATIONS")
    print("=" * 60)

    # Load and analyze data
    lda_data = load_example_data()
    analyze_topic_words_and_weights(lda_data)

    # Test word selection
    test_weighted_word_selection(lda_data)

    # Test current implementation
    current_results = test_current_implementation()

    # Test improved implementation
    improved_results = test_improved_mapping()

    print(f"\n=== SUMMARY COMPARISON ===")
    print(f"Current implementation results: {len(current_results)} mappings")
    print(f"Improved implementation results: {len(improved_results)} mappings")

    print("\nCurrent top 5:")
    current_sorted = sorted(current_results.items(), key=lambda x: x[1], reverse=True)
    for key, score in current_sorted[:5]:
        print(f"  {key}: {score}")

    print("\nImproved top 5:")
    improved_sorted = sorted(improved_results.items(), key=lambda x: x[1], reverse=True)
    for key, score in improved_sorted[:5]:
        print(f"  {key}: {score}")


if __name__ == "__main__":
    compare_results()
