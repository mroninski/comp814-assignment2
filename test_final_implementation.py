#!/usr/bin/env python3
"""
Final test for improved LDA taxonomy mapping implementation.
"""

import json
import sys
import os

# Add src to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from topic_extractor.topic_simplifying import (
    TopicTaxonomyMapper,
    map_lda_results_to_taxonomy,
)


def test_improved_implementation():
    """Test the improved implementation with example data."""
    print("=== TESTING IMPROVED LDA MAPPING ===")

    # Load example data
    with open("example.json", "r") as f:
        lda_data = json.load(f)

    # Initialize mapper
    mapper = TopicTaxonomyMapper()

    # Test improved implementation with different parameters
    print("\n--- Testing with top_n=10, min_threshold=20% ---")
    results = map_lda_results_to_taxonomy(mapper, lda_data, top_n=10)

    print("Results:")
    for key, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {key}: {score}")

    print(f"\nTotal results: {len(results)}")

    # Test with more permissive threshold
    print("\n--- Testing with top_n=15, more results ---")
    results2 = map_lda_results_to_taxonomy(
        mapper, lda_data, top_n=15, min_similarity=0.03
    )

    # Filter to show only >15% for readability
    filtered_results = {k: v for k, v in results2.items() if v >= 15.0}

    print("Results >= 15%:")
    for key, score in sorted(
        filtered_results.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {key}: {score}")

    print(f"\nFiltered results: {len(filtered_results)} (total: {len(results2)})")

    # Analysis
    print("\n=== ANALYSIS ===")
    print(
        "Topic 1 (watch_style) should map to entertainment/movies - ✓"
        if any("entertainment" in k for k in results.keys())
        else "❌"
    )
    print(
        "Topic 0 (fear_fear thing) should map to emotions/psychology - ✓"
        if any("emotions" in k for k in results.keys())
        else "❌"
    )

    return results


if __name__ == "__main__":
    test_improved_implementation()
