"""
Demographic Topic Analysis Module - Academic LDA Analysis

This module provides comprehensive analysis capabilities for extracting and analyzing
topics across different demographic segments with academic-style reporting.
"""

import json
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

warnings.filterwarnings("ignore")


class LDADemographicTopicAnalyzer:
    """
    Academic analyzer for extracting demographic-based topic insights from LDA blog data.

    This class focuses on statistical analysis and comparison between demographic groups,
    particularly gender-based comparisons, and extracts sample text for dominant topics.
    """

    def __init__(self, output_dir: str = ".data/lda_analysis_results"):
        """
        Initialize the analyzer with output directory configuration.

        Args:
            output_dir: Directory to save analysis results and visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define demographic groups for academic analysis
        self.demographic_groups = {
            "gender_comparison": {
                "name": "Gender Comparison Analysis",
                "segments": {
                    "male": {
                        "name": "Male",
                        "filter": lambda df: df.filter(
                            pl.col("gender").str.to_lowercase() == "male"
                        ),
                    },
                    "female": {
                        "name": "Female",
                        "filter": lambda df: df.filter(
                            pl.col("gender").str.to_lowercase() == "female"
                        ),
                    },
                },
            },
            "age_groups": {
                "name": "Age Group Analysis",
                "segments": {
                    "under_20": {
                        "name": "Under 20",
                        "filter": lambda df: df.filter(
                            pl.col("age").cast(pl.Int32) < 20
                        ),
                    },
                    "20_to_30": {
                        "name": "20-30",
                        "filter": lambda df: df.filter(
                            (pl.col("age").cast(pl.Int32) >= 20)
                            & (pl.col("age").cast(pl.Int32) < 30)
                        ),
                    },
                    "30_to_40": {
                        "name": "30-40",
                        "filter": lambda df: df.filter(
                            (pl.col("age").cast(pl.Int32) >= 30)
                            & (pl.col("age").cast(pl.Int32) < 40)
                        ),
                    },
                    "40_plus": {
                        "name": "40+",
                        "filter": lambda df: df.filter(
                            pl.col("age").cast(pl.Int32) >= 40
                        ),
                    },
                },
            },
            "student_comparison": {
                "name": "Student vs Non-Student Analysis",
                "segments": {
                    "students": {
                        "name": "Students",
                        "filter": lambda df: df.filter(
                            pl.col("industry")
                            .str.to_lowercase()
                            .str.contains("student")
                        ),
                    },
                    "non_students": {
                        "name": "Non-Students",
                        "filter": lambda df: df.filter(
                            ~pl.col("industry")
                            .str.to_lowercase()
                            .str.contains("student")
                        ),
                    },
                },
            },
            "industry_analysis": {
                "name": "Industry-Based Analysis",
                "segments": {
                    "technology": {
                        "name": "Technology",
                        "filter": lambda df: df.filter(
                            pl.col("industry")
                            .str.to_lowercase()
                            .str.contains("technology")
                            | pl.col("industry")
                            .str.to_lowercase()
                            .str.contains("computer")
                            | pl.col("industry")
                            .str.to_lowercase()
                            .str.contains("software")
                            | pl.col("industry")
                            .str.to_lowercase()
                            .str.contains("internet")
                        ),
                    },
                    "education": {
                        "name": "Education",
                        "filter": lambda df: df.filter(
                            pl.col("industry")
                            .str.to_lowercase()
                            .str.contains("education")
                            | pl.col("industry")
                            .str.to_lowercase()
                            .str.contains("teacher")
                            | pl.col("industry")
                            .str.to_lowercase()
                            .str.contains("school")
                        ),
                    },
                    "business": {
                        "name": "Business",
                        "filter": lambda df: df.filter(
                            pl.col("industry")
                            .str.to_lowercase()
                            .str.contains("business")
                            | pl.col("industry")
                            .str.to_lowercase()
                            .str.contains("marketing")
                            | pl.col("industry")
                            .str.to_lowercase()
                            .str.contains("finance")
                            | pl.col("industry")
                            .str.to_lowercase()
                            .str.contains("sales")
                        ),
                    },
                    "healthcare": {
                        "name": "Healthcare",
                        "filter": lambda df: df.filter(
                            pl.col("industry").str.to_lowercase().str.contains("health")
                            | pl.col("industry")
                            .str.to_lowercase()
                            .str.contains("medical")
                            | pl.col("industry")
                            .str.to_lowercase()
                            .str.contains("nurse")
                            | pl.col("industry")
                            .str.to_lowercase()
                            .str.contains("doctor")
                        ),
                    },
                },
            },
            "age_gender_intersection": {
                "name": "Age-Gender Intersection Analysis",
                "segments": {
                    "young_male": {
                        "name": "Young Male (Under 25)",
                        "filter": lambda df: df.filter(
                            (pl.col("age").cast(pl.Int32) < 25)
                            & (pl.col("gender").str.to_lowercase() == "male")
                        ),
                    },
                    "young_female": {
                        "name": "Young Female (Under 25)",
                        "filter": lambda df: df.filter(
                            (pl.col("age").cast(pl.Int32) < 25)
                            & (pl.col("gender").str.to_lowercase() == "female")
                        ),
                    },
                    "mature_male": {
                        "name": "Mature Male (25+)",
                        "filter": lambda df: df.filter(
                            (pl.col("age").cast(pl.Int32) >= 25)
                            & (pl.col("gender").str.to_lowercase() == "male")
                        ),
                    },
                    "mature_female": {
                        "name": "Mature Female (25+)",
                        "filter": lambda df: df.filter(
                            (pl.col("age").cast(pl.Int32) >= 25)
                            & (pl.col("gender").str.to_lowercase() == "female")
                        ),
                    },
                },
            },
            "student_gender_intersection": {
                "name": "Student-Gender Intersection Analysis",
                "segments": {
                    "male_students": {
                        "name": "Male Students",
                        "filter": lambda df: df.filter(
                            (
                                pl.col("industry")
                                .str.to_lowercase()
                                .str.contains("student")
                            )
                            & (pl.col("gender").str.to_lowercase() == "male")
                        ),
                    },
                    "female_students": {
                        "name": "Female Students",
                        "filter": lambda df: df.filter(
                            (
                                pl.col("industry")
                                .str.to_lowercase()
                                .str.contains("student")
                            )
                            & (pl.col("gender").str.to_lowercase() == "female")
                        ),
                    },
                    "male_professionals": {
                        "name": "Male Professionals",
                        "filter": lambda df: df.filter(
                            (
                                ~pl.col("industry")
                                .str.to_lowercase()
                                .str.contains("student")
                            )
                            & (pl.col("gender").str.to_lowercase() == "male")
                        ),
                    },
                    "female_professionals": {
                        "name": "Female Professionals",
                        "filter": lambda df: df.filter(
                            (
                                ~pl.col("industry")
                                .str.to_lowercase()
                                .str.contains("student")
                            )
                            & (pl.col("gender").str.to_lowercase() == "female")
                        ),
                    },
                },
            },
        }

    def analyze_lda_demographics(
        self,
        dataframe: pl.DataFrame,
        top_n_topics: int = 10,
        extract_clauses: bool = True,
    ) -> Dict[str, Any]:
        """
        Comprehensive LDA demographic topic analysis with academic focus on major/minor categories only.

        Args:
            dataframe: Input dataframe with blog data and extracted topics
            top_n_topics: Number of top topics to extract per demographic
            extract_clauses: Whether to extract sample clauses for dominant topics

        Returns:
            Dictionary containing comprehensive analysis results
        """

        # Focus only on taxonomy classification (major/minor categories)
        topic_columns = ["lda_taxonomy_classification"]

        print("Starting LDA Demographic Topic Analysis...")
        print(f"Dataset size: {dataframe.height} records")
        print(f"Focusing on major/minor categories only: {topic_columns}")

        if dataframe.is_empty():
            raise ValueError("Input dataframe is empty")

        results = {
            "metadata": {
                "total_records": dataframe.height,
                "analysis_timestamp": datetime.now().isoformat(),
                "topic_columns_analyzed": topic_columns,
                "demographic_groups": list(self.demographic_groups.keys()),
                "analysis_focus": "Major/Minor Categories Only (Taxonomy Classification)",
            },
            "demographic_analyses": {},
            "comprehensive_comparisons": {},
            "dominant_topic_clauses": {},
            "statistical_summary": {},
        }

        # Create demographic distribution summary
        print("Analyzing demographic distributions...")
        results["statistical_summary"] = self._create_demographic_statistics(dataframe)

        # Analyze each demographic group
        for group_id, group_config in self.demographic_groups.items():
            print(f"Analyzing demographic group: {group_config['name']}")

            group_results = self._analyze_demographic_group(
                dataframe, group_id, group_config, topic_columns, top_n_topics
            )

            results["demographic_analyses"][group_id] = group_results

            # Create gender comparisons if this is a gender-related group
            if (
                "gender" in group_id.lower()
                or len([
                    s for s in group_config["segments"] if "male" in s or "female" in s
                ])
                >= 2
            ):
                results["comprehensive_comparisons"][group_id] = (
                    self._create_gender_comparison(group_results, topic_columns)
                )

        # Extract clauses for dominant topics if requested
        if extract_clauses:
            print("Extracting clauses for dominant topics...")
            results["dominant_topic_clauses"] = self._extract_dominant_topic_clauses(
                dataframe, results["demographic_analyses"], topic_columns
            )

        # Create academic visualizations
        self._create_academic_visualizations(results, topic_columns)

        # Save comprehensive results
        self._save_academic_results(results)

        print(f"Analysis complete. Results saved to: {self.output_dir}")
        return results

    def _create_demographic_statistics(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Create statistical summary of demographic distributions."""

        stats = {
            "total_records": df.height,
            "gender_distribution": {},
            "age_distribution": {},
            "data_quality": {},
        }

        # Gender distribution
        try:
            gender_counts = (
                df.group_by("gender").agg(pl.count().alias("count")).to_pandas()
            )
            stats["gender_distribution"] = {
                row["gender"]: int(row["count"]) for _, row in gender_counts.iterrows()
            }
        except Exception as e:
            stats["gender_distribution"] = {"error": str(e)}

        # Age distribution
        try:
            age_stats = (
                df.select([
                    pl.col("age").cast(pl.Int32).min().alias("min_age"),
                    pl.col("age").cast(pl.Int32).max().alias("max_age"),
                    pl.col("age").cast(pl.Int32).mean().alias("mean_age"),
                    pl.col("age").cast(pl.Int32).median().alias("median_age"),
                ])
                .to_pandas()
                .iloc[0]
                .to_dict()
            )

            stats["age_distribution"] = {
                k: float(v) if pd.notna(v) else None for k, v in age_stats.items()
            }
        except Exception as e:
            stats["age_distribution"] = {"error": str(e)}

        return stats

    def _analyze_demographic_group(
        self,
        df: pl.DataFrame,
        group_id: str,
        group_config: Dict,
        topic_columns: List[str],
        top_n: int,
    ) -> Dict[str, Any]:
        """Analyze a single demographic group with statistical focus."""

        group_results = {
            "group_name": group_config["name"],
            "segments": {},
            "group_summary": {},
        }

        segments_with_data = []

        for segment_id, segment_config in group_config["segments"].items():
            try:
                filtered_df = segment_config["filter"](df)
                segment_size = filtered_df.height

                if segment_size == 0:
                    print(f"  No data for {segment_config['name']}")
                    continue

                print(f"  Analyzing {segment_config['name']}: {segment_size} records")

                # Analyze topics for this segment
                segment_analysis = self._analyze_segment_topics(
                    filtered_df, segment_config["name"], topic_columns, top_n
                )
                segment_analysis["segment_size"] = segment_size
                segment_analysis["segment_proportion"] = segment_size / df.height

                group_results["segments"][segment_id] = segment_analysis
                segments_with_data.append({
                    "id": segment_id,
                    "name": segment_config["name"],
                    "size": segment_size,
                    "analysis": segment_analysis,
                    "dataframe": filtered_df,  # Keep for clause extraction
                })

            except Exception as e:
                print(f"  Error analyzing {segment_config['name']}: {str(e)}")
                continue

        if segments_with_data:
            group_results["group_summary"] = self._create_group_summary(
                segments_with_data
            )

        group_results["segments_with_data"] = segments_with_data
        return group_results

    def _analyze_segment_topics(
        self, df: pl.DataFrame, segment_name: str, topic_columns: List[str], top_n: int
    ) -> Dict[str, Any]:
        """Analyze topics for a single demographic segment with academic metrics focusing on unique users."""

        segment_results = {}

        for col in topic_columns:
            topic_user_sets = defaultdict(set)  # Track unique users per topic
            major_category_user_sets = defaultdict(set)
            subcategory_user_sets = defaultdict(set)

            # Process each row and track which users discussed which topics
            df_pandas = df.to_pandas()
            for idx, row in df_pandas.iterrows():
                topic_data = row[col]

                if pd.isna(topic_data) or topic_data is None:
                    continue

                try:
                    # Handle different data formats
                    if isinstance(topic_data, str):
                        if topic_data.startswith("{") or topic_data.startswith("["):
                            topics_data = json.loads(topic_data)
                        else:
                            topics_data = [
                                topic.strip() for topic in topic_data.split(",")
                            ]
                    else:
                        topics_data = topic_data

                    # Extract topics based on data structure
                    if isinstance(topics_data, dict):
                        if col == "lda_taxonomy_classification":
                            for taxonomy_key, score in topics_data.items():
                                if (
                                    isinstance(taxonomy_key, str)
                                    and ":" in taxonomy_key
                                ):
                                    parts = taxonomy_key.split(":", 1)
                                    if len(parts) == 2:
                                        major, subcategory = parts
                                        major = major.strip()
                                        subcategory = subcategory.strip()

                                        # Add user to topic sets (unique users only)
                                        topic_user_sets[taxonomy_key].add(idx)
                                        major_category_user_sets[major].add(idx)
                                        subcategory_user_sets[subcategory].add(idx)
                                else:
                                    topic_user_sets[taxonomy_key].add(idx)
                        else:
                            for key, value in topics_data.items():
                                if isinstance(value, (list, tuple)):
                                    for item in value:
                                        topic_user_sets[str(item)].add(idx)
                                else:
                                    topic_user_sets[key].add(idx)
                    elif isinstance(topics_data, (list, tuple)):
                        if col == "lda_taxonomy_classification":
                            for item in topics_data:
                                if isinstance(item, str) and ":" in item:
                                    parts = item.split(":", 1)
                                    if len(parts) == 2:
                                        major, subcategory = parts
                                        major = major.strip()
                                        subcategory = subcategory.strip()

                                        topic_user_sets[item].add(idx)
                                        major_category_user_sets[major].add(idx)
                                        subcategory_user_sets[subcategory].add(idx)
                                else:
                                    topic_user_sets[str(item)].add(idx)
                        else:
                            for item in topics_data:
                                topic_user_sets[str(item)].add(idx)
                    elif isinstance(topics_data, str):
                        topic_user_sets[topics_data].add(idx)

                except (json.JSONDecodeError, TypeError, AttributeError):
                    continue

            # Convert user sets to counts (number of unique users per topic)
            topic_counter = {
                topic: len(user_set) for topic, user_set in topic_user_sets.items()
            }
            major_category_counter = {
                category: len(user_set)
                for category, user_set in major_category_user_sets.items()
            }
            subcategory_counter = {
                subcategory: len(user_set)
                for subcategory, user_set in subcategory_user_sets.items()
            }

            # Calculate academic metrics based on unique users
            top_topics = sorted(
                topic_counter.items(), key=lambda x: x[1], reverse=True
            )[:top_n]
            total_unique_topics = len(topic_counter)
            total_users_with_topics = (
                len(set().union(*topic_user_sets.values())) if topic_user_sets else 0
            )

            result = {
                "top_topics": top_topics,
                "total_unique_topics": total_unique_topics,
                "total_topic_instances": sum(
                    topic_counter.values()
                ),  # This is now unique user-topic combinations
                "records_with_topics": total_users_with_topics,
                "topic_coverage": total_users_with_topics / df.height
                if df.height > 0
                else 0,
                "avg_topics_per_record": sum(
                    len(user_set) for user_set in topic_user_sets.values()
                )
                / df.height
                if df.height > 0
                else 0,
                "topic_concentration": (top_topics[0][1] / total_users_with_topics)
                if top_topics and total_users_with_topics > 0
                else 0,
                "full_distribution": topic_counter,
            }

            # Add taxonomy-specific analysis
            if col == "lda_taxonomy_classification" and (
                major_category_counter or subcategory_counter
            ):
                result["taxonomy_analysis"] = {
                    "major_categories": {
                        "top_categories": sorted(
                            major_category_counter.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )[:top_n],
                        "total_unique_categories": len(major_category_counter),
                        "total_category_instances": sum(
                            major_category_counter.values()
                        ),
                        "full_distribution": major_category_counter,
                    },
                    "subcategories": {
                        "top_subcategories": sorted(
                            subcategory_counter.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )[:top_n],
                        "total_unique_subcategories": len(subcategory_counter),
                        "total_subcategory_instances": sum(
                            subcategory_counter.values()
                        ),
                        "full_distribution": subcategory_counter,
                    },
                }

            segment_results[col] = result

        return segment_results

    def _create_gender_comparison(
        self, group_results: Dict, topic_columns: List[str]
    ) -> Dict[str, Any]:
        """Create separate academic charts for each demographic difference in major/minor categories."""

        comparison = {
            "category_differences": {},
            "major_category_differences": {},
            "subcategory_differences": {},
            "statistical_tests": {},
            "notable_differences": [],
        }

        # Get all segments for comparison
        segments = list(group_results["segments"].items())

        if len(segments) < 2:
            return {"error": "Need at least 2 segments for comparison"}

        # For binary comparisons (like male/female, student/non-student)
        if len(segments) == 2:
            segment1_id, segment1_data = segments[0]
            segment2_id, segment2_data = segments[1]

            segment1_name = segment1_id.replace("_", " ").title()
            segment2_name = segment2_id.replace("_", " ").title()

            for col in topic_columns:
                if col in segment1_data and col in segment2_data:
                    # Full taxonomy comparison
                    seg1_topics = dict(segment1_data[col]["top_topics"])
                    seg2_topics = dict(segment2_data[col]["top_topics"])

                    all_topics = set(seg1_topics.keys()) | set(seg2_topics.keys())
                    topic_differences = {}

                    for topic in all_topics:
                        seg1_count = seg1_topics.get(topic, 0)
                        seg2_count = seg2_topics.get(topic, 0)

                        seg1_rate = seg1_count / segment1_data["segment_size"]
                        seg2_rate = seg2_count / segment2_data["segment_size"]

                        if seg1_rate + seg2_rate > 0:
                            difference_ratio = (seg1_rate - seg2_rate) / (
                                seg1_rate + seg2_rate
                            )
                            topic_differences[topic] = {
                                f"{segment1_name.lower()}_count": seg1_count,
                                f"{segment2_name.lower()}_count": seg2_count,
                                f"{segment1_name.lower()}_rate": seg1_rate,
                                f"{segment2_name.lower()}_rate": seg2_rate,
                                "difference_ratio": difference_ratio,
                                "absolute_difference": abs(seg1_count - seg2_count),
                            }

                    # Find most notable differences
                    notable = sorted(
                        topic_differences.items(),
                        key=lambda x: abs(x[1]["difference_ratio"]),
                        reverse=True,
                    )[:15]  # Top 15 for better analysis

                    comparison["category_differences"][col] = {
                        "all_differences": topic_differences,
                        "most_notable": notable,
                        f"{segment1_name.lower()}_dominant": [
                            t for t, d in notable if d["difference_ratio"] > 0
                        ],
                        f"{segment2_name.lower()}_dominant": [
                            t for t, d in notable if d["difference_ratio"] < 0
                        ],
                    }

                    # Separate major categories and subcategories analysis
                    if (
                        col == "lda_taxonomy_classification"
                        and "taxonomy_analysis" in segment1_data[col]
                    ):
                        # Major categories comparison
                        seg1_major = dict(
                            segment1_data[col]["taxonomy_analysis"]["major_categories"][
                                "top_categories"
                            ]
                        )
                        seg2_major = dict(
                            segment2_data[col]["taxonomy_analysis"]["major_categories"][
                                "top_categories"
                            ]
                        )

                        major_comparison = self._compare_category_sets(
                            seg1_major,
                            seg2_major,
                            segment1_data["segment_size"],
                            segment2_data["segment_size"],
                            segment1_name,
                            segment2_name,
                        )
                        comparison["major_category_differences"][col] = major_comparison

                        # Subcategories comparison
                        seg1_sub = dict(
                            segment1_data[col]["taxonomy_analysis"]["subcategories"][
                                "top_subcategories"
                            ]
                        )
                        seg2_sub = dict(
                            segment2_data[col]["taxonomy_analysis"]["subcategories"][
                                "top_subcategories"
                            ]
                        )

                        sub_comparison = self._compare_category_sets(
                            seg1_sub,
                            seg2_sub,
                            segment1_data["segment_size"],
                            segment2_data["segment_size"],
                            segment1_name,
                            segment2_name,
                        )
                        comparison["subcategory_differences"][col] = sub_comparison

        return comparison

    def _compare_category_sets(
        self, set1: Dict, set2: Dict, size1: int, size2: int, name1: str, name2: str
    ) -> Dict:
        """Helper method to compare two sets of categories."""

        all_categories = set(set1.keys()) | set(set2.keys())
        category_differences = {}

        for category in all_categories:
            count1 = set1.get(category, 0)
            count2 = set2.get(category, 0)

            rate1 = count1 / size1 if size1 > 0 else 0
            rate2 = count2 / size2 if size2 > 0 else 0

            if rate1 + rate2 > 0:
                difference_ratio = (rate1 - rate2) / (rate1 + rate2)
                category_differences[category] = {
                    f"{name1.lower()}_count": count1,
                    f"{name2.lower()}_count": count2,
                    f"{name1.lower()}_rate": rate1,
                    f"{name2.lower()}_rate": rate2,
                    "difference_ratio": difference_ratio,
                    "absolute_difference": abs(count1 - count2),
                }

        # Find most notable differences
        notable = sorted(
            category_differences.items(),
            key=lambda x: abs(x[1]["difference_ratio"]),
            reverse=True,
        )[:10]

        return {
            "all_differences": category_differences,
            "most_notable": notable,
            f"{name1.lower()}_dominant": [
                t for t, d in notable if d["difference_ratio"] > 0
            ],
            f"{name2.lower()}_dominant": [
                t for t, d in notable if d["difference_ratio"] < 0
            ],
        }

    def _extract_dominant_topic_clauses(
        self, df: pl.DataFrame, demographic_analyses: Dict, topic_columns: List[str]
    ) -> Dict[str, Any]:
        """Extract sample clauses containing the 2 dominant topics for each demographic."""

        dominant_clauses = {}

        for group_id, group_data in demographic_analyses.items():
            dominant_clauses[group_id] = {}

            for segment_id, segment_data in group_data["segments"].items():
                if "segments_with_data" in group_data:
                    # Find the corresponding dataframe
                    segment_df = None
                    for seg_with_data in group_data["segments_with_data"]:
                        if seg_with_data["id"] == segment_id:
                            segment_df = seg_with_data["dataframe"]
                            break

                    if segment_df is None:
                        continue

                    segment_clauses = {}

                    for col in topic_columns:
                        if col in segment_data:
                            top_topics = segment_data[col]["top_topics"][
                                :2
                            ]  # Get top 2 topics

                            for topic, count in top_topics:
                                clauses = self._find_clauses_with_topic(
                                    segment_df, col, topic, max_clauses=5
                                )
                                segment_clauses[f"{col}_{topic}"] = {
                                    "topic": topic,
                                    "count": count,
                                    "sample_clauses": clauses,
                                }

                    dominant_clauses[group_id][segment_id] = segment_clauses

        return dominant_clauses

    def _find_clauses_with_topic(
        self,
        df: pl.DataFrame,
        topic_column: str,
        target_topic: str,
        max_clauses: int = 5,
    ) -> List[str]:
        """Find sample text clauses that contain the specified topic."""

        clauses = []

        try:
            # Get records that contain this topic
            for row in df.select([topic_column, "content"]).to_pandas().itertuples():
                topic_data = getattr(row, topic_column.replace("-", "_"))
                text_data = row.content

                if pd.isna(topic_data) or pd.isna(text_data):
                    continue

                # Check if this record contains the target topic
                contains_topic = False

                try:
                    if isinstance(topic_data, str):
                        if topic_data.startswith("{") or topic_data.startswith("["):
                            parsed_data = json.loads(topic_data)
                        else:
                            parsed_data = [t.strip() for t in topic_data.split(",")]
                    else:
                        parsed_data = topic_data

                    # Check if target topic is in this record's topics
                    if isinstance(parsed_data, dict):
                        contains_topic = target_topic in parsed_data
                    elif isinstance(parsed_data, (list, tuple)):
                        contains_topic = target_topic in parsed_data
                    elif isinstance(parsed_data, str):
                        contains_topic = target_topic == parsed_data

                except (json.JSONDecodeError, TypeError):
                    continue

                if contains_topic and len(clauses) < max_clauses:
                    # Extract a relevant sentence or clause from the text
                    text_sentences = str(text_data).split(". ")

                    # Look for sentences that might relate to the topic
                    topic_words = target_topic.lower().split()
                    best_sentence = ""
                    max_matches = 0

                    for sentence in text_sentences:
                        sentence_lower = sentence.lower()
                        matches = sum(
                            1 for word in topic_words if word in sentence_lower
                        )
                        if matches > max_matches:
                            max_matches = matches
                            best_sentence = sentence.strip()

                    if (
                        best_sentence and len(best_sentence) > 20
                    ):  # Ensure meaningful length
                        clauses.append(
                            best_sentence[:200] + "..."
                            if len(best_sentence) > 200
                            else best_sentence
                        )

        except Exception as e:
            print(f"Error extracting clauses for topic {target_topic}: {str(e)}")

        return clauses

    def _create_academic_visualizations(self, results: Dict, topic_columns: List[str]):
        """Create academic-style visualizations focusing on individual category comparisons."""

        print("Creating individual academic visualizations for each category...")

        # Set academic style
        plt.style.use("default")
        sns.set_style("whitegrid")
        sns.set_palette("Set2")

        # Create individual demographic comparison charts
        self._create_gender_comparison_charts(results, topic_columns)

        # Create demographic distribution charts
        self._create_demographic_distribution_charts(results)

        # Create comprehensive summary dashboard
        self._create_comprehensive_summary_dashboard(results, topic_columns)

        print("All individual category visualizations completed.")

    def _create_gender_comparison_charts(self, results: Dict, topic_columns: List[str]):
        """Create separate academic charts for each demographic difference in major/minor categories."""

        gender_data = results.get("comprehensive_comparisons", {})

        for group_id, comparison_data in gender_data.items():
            if "category_differences" not in comparison_data:
                continue

            for col in topic_columns:
                if col not in comparison_data["category_differences"]:
                    continue

                topic_diffs = comparison_data["category_differences"][col]
                notable_diffs = topic_diffs.get("most_notable", [])

                if not notable_diffs:
                    continue

                # Extract segment names dynamically
                first_diff = notable_diffs[0][1]
                segment_names = []
                for key in first_diff.keys():
                    if key.endswith("_count"):
                        segment_names.append(key.replace("_count", "").title())

                if len(segment_names) != 2:
                    continue

                seg1_name, seg2_name = segment_names

                # Create individual charts for each category (no truncation)
                for i, (topic, diff_data) in enumerate(notable_diffs):
                    # Individual category comparison chart
                    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

                    # Full category name as title
                    fig.suptitle(
                        f"{topic}\n{group_id.replace('_', ' ').title()} Comparison",
                        fontsize=14,
                        fontweight="bold",
                        wrap=True,
                    )

                    # Use percentages instead of raw counts
                    percentages = [
                        diff_data[f"{seg1_name.lower()}_rate"] * 100,
                        diff_data[f"{seg2_name.lower()}_rate"] * 100,
                    ]
                    labels = [seg1_name, seg2_name]
                    colors = ["steelblue", "lightcoral"]

                    bars = ax.bar(
                        labels, percentages, color=colors, alpha=0.8, width=0.6
                    )
                    ax.set_ylabel("Percentage of Users (%)", fontsize=12)
                    ax.set_title(
                        "Percentage of Users Discussing This Topic", fontsize=12
                    )
                    ax.grid(axis="y", alpha=0.3)
                    ax.set_ylim(
                        0, max(percentages) * 1.1
                    )  # Set appropriate y-axis limit

                    # Add percentage labels on bars
                    for bar, percentage in zip(bars, percentages):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 1,
                            f"{percentage:.1f}%",
                            ha="center",
                            va="bottom",
                            fontweight="bold",
                            fontsize=11,
                        )

                    # Add difference ratio and raw counts information
                    diff_ratio = diff_data["difference_ratio"]
                    dominant = seg1_name if diff_ratio > 0 else seg2_name
                    raw_counts = f"Raw counts: {seg1_name} ({diff_data[f'{seg1_name.lower()}_count']}), {seg2_name} ({diff_data[f'{seg2_name.lower()}_count']})"

                    ax.text(
                        0.5,
                        0.95,
                        f"Difference Ratio: {diff_ratio:.3f}\nDominant: {dominant}\n{raw_counts}",
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7
                        ),
                        fontsize=9,
                    )

                    plt.tight_layout()

                    # Clean filename with full category name
                    clean_topic = (
                        topic.replace(":", "_").replace("/", "_").replace(" ", "_")
                    )
                    filename = f"category_{group_id}_{clean_topic}_{col}.png"
                    filepath = self.output_dir / filename
                    plt.savefig(filepath, dpi=300, bbox_inches="tight")
                    plt.close()

                    print(f"Saved individual category chart: {filename}")

                # Create major categories chart if available
                if (
                    "major_category_differences" in comparison_data
                    and col in comparison_data["major_category_differences"]
                ):
                    self._create_individual_major_category_charts(
                        group_id,
                        comparison_data["major_category_differences"][col],
                        col,
                        seg1_name,
                        seg2_name,
                    )

                # Create subcategories chart if available
                if (
                    "subcategory_differences" in comparison_data
                    and col in comparison_data["subcategory_differences"]
                ):
                    self._create_individual_subcategory_charts(
                        group_id,
                        comparison_data["subcategory_differences"][col],
                        col,
                        seg1_name,
                        seg2_name,
                    )

    def _create_individual_major_category_charts(
        self, group_id: str, major_data: Dict, col: str, seg1_name: str, seg2_name: str
    ):
        """Create individual charts for each major category comparison."""

        notable_diffs = major_data.get("most_notable", [])
        if not notable_diffs:
            return

        for category, diff_data in notable_diffs:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            fig.suptitle(
                f"Major Category: {category}\n{group_id.replace('_', ' ').title()} Comparison",
                fontsize=14,
                fontweight="bold",
            )

            # Use percentages instead of raw counts
            percentages = [
                diff_data[f"{seg1_name.lower()}_rate"] * 100,
                diff_data[f"{seg2_name.lower()}_rate"] * 100,
            ]
            labels = [seg1_name, seg2_name]
            colors = ["steelblue", "lightcoral"]

            bars = ax.bar(labels, percentages, color=colors, alpha=0.8, width=0.6)
            ax.set_ylabel("Percentage of Users (%)", fontsize=12)
            ax.set_title(
                "Percentage of Users Discussing This Major Category", fontsize=12
            )
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(0, max(percentages) * 1.1)  # Set appropriate y-axis limit

            # Add percentage labels on bars
            for bar, percentage in zip(bars, percentages):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(percentages) * 0.02,
                    f"{percentage:.1f}%",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=11,
                )

            # Add difference ratio and raw counts information
            diff_ratio = diff_data["difference_ratio"]
            dominant = seg1_name if diff_ratio > 0 else seg2_name
            raw_counts = f"Raw counts: {seg1_name} ({diff_data[f'{seg1_name.lower()}_count']}), {seg2_name} ({diff_data[f'{seg2_name.lower()}_count']})"

            ax.text(
                0.5,
                0.95,
                f"Difference Ratio: {diff_ratio:.3f}\nDominant: {dominant}\n{raw_counts}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                fontsize=9,
            )

            plt.tight_layout()

            clean_category = category.replace(" ", "_").replace("/", "_")
            filename = f"major_category_{group_id}_{clean_category}_{col}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Saved major category chart: {filename}")

    def _create_individual_subcategory_charts(
        self, group_id: str, sub_data: Dict, col: str, seg1_name: str, seg2_name: str
    ):
        """Create individual charts for each subcategory comparison."""

        notable_diffs = sub_data.get("most_notable", [])
        if not notable_diffs:
            return

        for subcategory, diff_data in notable_diffs:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            fig.suptitle(
                f"Subcategory: {subcategory}\n{group_id.replace('_', ' ').title()} Comparison",
                fontsize=14,
                fontweight="bold",
            )

            # Use percentages instead of raw counts
            percentages = [
                diff_data[f"{seg1_name.lower()}_rate"] * 100,
                diff_data[f"{seg2_name.lower()}_rate"] * 100,
            ]
            labels = [seg1_name, seg2_name]
            colors = ["steelblue", "lightcoral"]

            bars = ax.bar(labels, percentages, color=colors, alpha=0.8, width=0.6)
            ax.set_ylabel("Percentage of Users (%)", fontsize=12)
            ax.set_title("Percentage of Users Discussing This Subcategory", fontsize=12)
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(0, max(percentages) * 1.1)  # Set appropriate y-axis limit

            # Add percentage labels on bars
            for bar, percentage in zip(bars, percentages):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(percentages) * 0.02,
                    f"{percentage:.1f}%",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=11,
                )

            # Add difference ratio and raw counts information
            diff_ratio = diff_data["difference_ratio"]
            dominant = seg1_name if diff_ratio > 0 else seg2_name
            raw_counts = f"Raw counts: {seg1_name} ({diff_data[f'{seg1_name.lower()}_count']}), {seg2_name} ({diff_data[f'{seg2_name.lower()}_count']})"

            ax.text(
                0.5,
                0.95,
                f"Difference Ratio: {diff_ratio:.3f}\nDominant: {dominant}\n{raw_counts}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                fontsize=9,
            )

            plt.tight_layout()

            clean_subcategory = (
                subcategory.replace(":", "_").replace(" ", "_").replace("/", "_")
            )
            filename = f"subcategory_{group_id}_{clean_subcategory}_{col}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Saved subcategory chart: {filename}")

    def _create_demographic_distribution_charts(self, results: Dict):
        """Create charts showing demographic distributions."""

        stats = results.get("statistical_summary", {})
        demographic_analyses = results.get("demographic_analyses", {})

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Demographic Distribution Analysis", fontsize=16, fontweight="bold"
        )

        # Gender distribution
        if (
            "gender_distribution" in stats
            and "error" not in stats["gender_distribution"]
        ):
            genders = list(stats["gender_distribution"].keys())
            counts = list(stats["gender_distribution"].values())
            ax1.pie(counts, labels=genders, autopct="%1.1f%%", startangle=90)
            ax1.set_title("Gender Distribution")
        else:
            ax1.text(0.5, 0.5, "No gender data available", ha="center", va="center")
            ax1.set_title("Gender Distribution - No Data")

        # Age distribution from actual demographic analysis
        age_group_data = demographic_analyses.get("age_groups", {})
        if age_group_data and "segments" in age_group_data:
            age_segments = []
            age_counts = []

            for segment_id, segment_data in age_group_data["segments"].items():
                segment_size = segment_data.get("segment_size", 0)
                if segment_size > 0:
                    # Map segment IDs to readable names
                    if segment_id == "under_20":
                        age_segments.append("Under 20")
                    elif segment_id == "20_to_30":
                        age_segments.append("20-30")
                    elif segment_id == "30_to_40":
                        age_segments.append("30-40")
                    elif segment_id == "40_plus":
                        age_segments.append("40+")
                    else:
                        age_segments.append(segment_id)
                    age_counts.append(segment_size)

            if age_segments and age_counts:
                bars = ax2.bar(age_segments, age_counts)
                ax2.set_title("Age Group Distribution")
                ax2.set_ylabel("Count")
                ax2.set_xlabel("Age Groups")
                # Add value labels on bars
                for bar, count in zip(bars, age_counts):
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.05,
                        str(count),
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )
            else:
                ax2.text(0.5, 0.5, "No age group data", ha="center", va="center")
                ax2.set_title("Age Distribution - No Data")
        else:
            ax2.text(0.5, 0.5, "No age data available", ha="center", va="center")
            ax2.set_title("Age Distribution - No Data")

        # Total records summary
        total_records = stats.get("total_records", 0)
        ax4.text(
            0.5,
            0.5,
            f"Total Records\n{total_records:,}",
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
        )
        ax4.set_title("Dataset Size")
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis("off")

        plt.tight_layout()

        filename = "academic_demographic_overview.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {filename}")

    def _create_comprehensive_summary_dashboard(
        self, results: Dict, topic_columns: List[str]
    ):
        """Create a comprehensive summary dashboard showing all demographics compared for the top topics in a single chart."""

        print("Creating comprehensive summary dashboard...")

        demographic_analyses = results.get("demographic_analyses", {})

        for col in topic_columns:
            # Collect all demographic segments and their top topics
            all_segments = []
            all_topics_set = set()

            for group_id, group_data in demographic_analyses.items():
                for segment_id, segment_data in group_data.get("segments", {}).items():
                    if col in segment_data and "top_topics" in segment_data[col]:
                        segment_info = {
                            "group_id": group_id,
                            "segment_id": segment_id,
                            "segment_name": f"{group_id.replace('_', ' ').title()}: {segment_id.replace('_', ' ').title()}",
                            "segment_size": segment_data.get("segment_size", 0),
                            "topic_data": dict(segment_data[col]["top_topics"]),
                            "full_distribution": segment_data[col].get(
                                "full_distribution", {}
                            ),
                        }
                        all_segments.append(segment_info)
                        # Collect all topics mentioned across segments
                        all_topics_set.update(segment_info["topic_data"].keys())

            if not all_segments:
                continue

            # Get the top 15 most frequently mentioned topics across all demographics
            topic_frequency = {}
            for topic in all_topics_set:
                topic_frequency[topic] = sum(
                    1 for seg in all_segments if topic in seg["topic_data"]
                )

            top_global_topics = sorted(
                topic_frequency.items(), key=lambda x: x[1], reverse=True
            )[:15]
            top_topics = [topic for topic, _ in top_global_topics]

            # Create the comprehensive comparison chart
            fig, ax = plt.subplots(1, 1, figsize=(20, 12))
            fig.suptitle(
                f"Comprehensive Demographic Comparison Dashboard\nTop Topics Across All Demographics",
                fontsize=16,
                fontweight="bold",
            )

            # Prepare data for visualization
            segment_names = [seg["segment_name"] for seg in all_segments]
            topic_matrix = []

            for topic in top_topics:
                topic_percentages = []
                for segment in all_segments:
                    user_count = segment["topic_data"].get(topic, 0)
                    percentage = (
                        (user_count / segment["segment_size"]) * 100
                        if segment["segment_size"] > 0
                        else 0
                    )
                    topic_percentages.append(percentage)
                topic_matrix.append(topic_percentages)

            # Create heatmap-style visualization
            topic_matrix = np.array(topic_matrix)

            # Create heatmap
            im = ax.imshow(topic_matrix, cmap="YlOrRd", aspect="auto")

            # Set ticks and labels
            ax.set_xticks(np.arange(len(segment_names)))
            ax.set_yticks(np.arange(len(top_topics)))
            ax.set_xticklabels(segment_names, rotation=45, ha="right", fontsize=10)
            ax.set_yticklabels(
                [
                    topic[:40] + "..." if len(topic) > 40 else topic
                    for topic in top_topics
                ],
                fontsize=10,
            )

            # Add percentage values on the heatmap
            for i in range(len(top_topics)):
                for j in range(len(segment_names)):
                    percentage = topic_matrix[i, j]
                    if percentage > 0:
                        text_color = "white" if percentage > 50 else "black"
                        ax.text(
                            j,
                            i,
                            f"{percentage:.1f}%",
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=8,
                            fontweight="bold",
                        )

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(
                "Percentage of Users (%)", rotation=270, labelpad=20, fontsize=12
            )

            # Add title and labels
            ax.set_xlabel("Demographic Segments", fontsize=12, fontweight="bold")
            ax.set_ylabel(
                "Topics (Major:Minor Categories)", fontsize=12, fontweight="bold"
            )

            # Add summary statistics
            total_segments = len(all_segments)
            total_topics = len(top_topics)
            avg_coverage = (
                np.mean(topic_matrix[topic_matrix > 0])
                if np.any(topic_matrix > 0)
                else 0
            )

            summary_text = f"""Summary Statistics:
• {total_segments} demographic segments analyzed
• {total_topics} top topics displayed
• Average coverage: {avg_coverage:.1f}% when topic is discussed
• Heat intensity shows percentage of users in each demographic discussing each topic"""

            ax.text(
                1.02,
                0.5,
                summary_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7),
            )

            plt.tight_layout()

            filename = f"comprehensive_dashboard_all_demographics_{col}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Saved comprehensive dashboard: {filename}")

            # Also create a detailed comparison table chart
            self._create_detailed_comparison_table_chart(all_segments, top_topics, col)

    def _create_detailed_comparison_table_chart(
        self, all_segments: List[Dict], top_topics: List[str], col: str
    ):
        """Create a detailed table-style chart showing exact percentages for all demographics."""

        fig, ax = plt.subplots(1, 1, figsize=(18, 14))
        fig.suptitle(
            f"Detailed Demographic Comparison Table\nPercentage of Users Discussing Each Topic",
            fontsize=16,
            fontweight="bold",
        )

        # Prepare table data
        table_data = []
        row_labels = []

        for topic in top_topics[:12]:  # Limit to top 12 for readability
            row = []
            for segment in all_segments:
                user_count = segment["topic_data"].get(topic, 0)
                percentage = (
                    (user_count / segment["segment_size"]) * 100
                    if segment["segment_size"] > 0
                    else 0
                )
                raw_count = segment["topic_data"].get(topic, 0)
                # Show both percentage and raw count
                cell_text = f"{percentage:.1f}%\n({raw_count})"
                row.append(cell_text)
            table_data.append(row)
            # Truncate topic name for display
            topic_display = topic[:30] + "..." if len(topic) > 30 else topic
            row_labels.append(topic_display)

        # Column labels (demographic segments)
        col_labels = [
            seg["segment_name"][:25] + "..."
            if len(seg["segment_name"]) > 25
            else seg["segment_name"]
            for seg in all_segments
        ]

        # Create table
        table = ax.table(
            cellText=table_data,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)

        # Color cells based on percentage values
        for i in range(len(top_topics[:12])):
            for j in range(len(all_segments)):
                user_count = all_segments[j]["topic_data"].get(top_topics[i], 0)
                percentage = (
                    (user_count / all_segments[j]["segment_size"]) * 100
                    if all_segments[j]["segment_size"] > 0
                    else 0
                )

                # Color intensity based on percentage
                if percentage > 80:
                    color = "#d73027"  # Dark red
                elif percentage > 60:
                    color = "#fc8d59"  # Orange red
                elif percentage > 40:
                    color = "#fee08b"  # Yellow
                elif percentage > 20:
                    color = "#e0f3f8"  # Light blue
                elif percentage > 0:
                    color = "#f7f7f7"  # Very light gray
                else:
                    color = "#ffffff"  # White

                table[(i + 1, j)].set_facecolor(color)

        # Style headers
        for j in range(len(all_segments)):
            table[(0, j)].set_facecolor("#4575b4")
            table[(0, j)].set_text_props(weight="bold", color="white")

        for i in range(len(top_topics[:12])):
            table[(i + 1, -1)].set_facecolor("#4575b4")
            table[(i + 1, -1)].set_text_props(weight="bold", color="white")

        ax.axis("off")

        # Add legend
        legend_text = """Color Legend:
🔴 >80%: Very High Engagement
🟠 60-80%: High Engagement  
🟡 40-60%: Medium Engagement
🔵 20-40%: Low Engagement
⚪ 0-20%: Very Low Engagement

Format: Percentage (Raw Count)"""

        ax.text(
            1.02,
            0.5,
            legend_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        )

        plt.tight_layout()

        filename = f"detailed_comparison_table_all_demographics_{col}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved detailed comparison table: {filename}")

    def _save_academic_results(self, results: Dict):
        """Save comprehensive academic results to files."""

        print("Saving academic analysis results...")

        # Save main results as JSON
        results_copy = results.copy()
        # Remove dataframes which can't be serialized
        for group_id, group_data in results_copy.get(
            "demographic_analyses", {}
        ).items():
            if "segments_with_data" in group_data:
                for segment in group_data["segments_with_data"]:
                    if "dataframe" in segment:
                        del segment["dataframe"]

        with open(self.output_dir / "academic_lda_analysis_results.json", "w") as f:
            json.dump(results_copy, f, indent=2, default=str)

        # Save gender comparison table
        self._save_gender_comparison_table(results)

        # Save dominant topic clauses
        self._save_dominant_clauses_table(results)

        # Save demographic summary table
        self._save_demographic_summary_table(results)

    def _save_gender_comparison_table(self, results: Dict):
        """Save comprehensive demographic comparison analysis as CSV."""

        comprehensive_comparisons = results.get("comprehensive_comparisons", {})
        comparison_rows = []

        for group_id, comparison_data in comprehensive_comparisons.items():
            # Save full category differences
            category_diffs = comparison_data.get("category_differences", {})

            for col, col_data in category_diffs.items():
                notable_diffs = col_data.get("most_notable", [])

                for topic, diff_data in notable_diffs:
                    # Extract demographic names dynamically
                    segment_names = []
                    for key in diff_data.keys():
                        if key.endswith("_count"):
                            segment_names.append(key.replace("_count", ""))

                    if len(segment_names) == 2:
                        seg1_name, seg2_name = segment_names
                        comparison_rows.append({
                            "demographic_group": group_id,
                            "topic_column": col,
                            "category": topic,
                            f"{seg1_name}_count": diff_data[f"{seg1_name}_count"],
                            f"{seg2_name}_count": diff_data[f"{seg2_name}_count"],
                            f"{seg1_name}_rate": round(
                                diff_data[f"{seg1_name}_rate"], 4
                            ),
                            f"{seg2_name}_rate": round(
                                diff_data[f"{seg2_name}_rate"], 4
                            ),
                            "difference_ratio": round(diff_data["difference_ratio"], 4),
                            "absolute_difference": diff_data["absolute_difference"],
                            "dominant_demographic": seg1_name
                            if diff_data["difference_ratio"] > 0
                            else seg2_name,
                        })

        if comparison_rows:
            df = pd.DataFrame(comparison_rows)
            df.to_csv(
                self.output_dir / "academic_demographic_comparison_analysis.csv",
                index=False,
            )
            print("Saved: academic_demographic_comparison_analysis.csv")

        # Save major categories comparison
        self._save_major_categories_comparison(comprehensive_comparisons)

        # Save subcategories comparison
        self._save_subcategories_comparison(comprehensive_comparisons)

    def _save_major_categories_comparison(self, comprehensive_comparisons: Dict):
        """Save major categories comparison analysis."""

        major_rows = []

        for group_id, comparison_data in comprehensive_comparisons.items():
            major_diffs = comparison_data.get("major_category_differences", {})

            for col, col_data in major_diffs.items():
                notable_diffs = col_data.get("most_notable", [])

                for category, diff_data in notable_diffs:
                    # Extract demographic names dynamically
                    segment_names = []
                    for key in diff_data.keys():
                        if key.endswith("_count"):
                            segment_names.append(key.replace("_count", ""))

                    if len(segment_names) == 2:
                        seg1_name, seg2_name = segment_names
                        major_rows.append({
                            "demographic_group": group_id,
                            "major_category": category,
                            f"{seg1_name}_count": diff_data[f"{seg1_name}_count"],
                            f"{seg2_name}_count": diff_data[f"{seg2_name}_count"],
                            f"{seg1_name}_rate": round(
                                diff_data[f"{seg1_name}_rate"], 4
                            ),
                            f"{seg2_name}_rate": round(
                                diff_data[f"{seg2_name}_rate"], 4
                            ),
                            "difference_ratio": round(diff_data["difference_ratio"], 4),
                            "absolute_difference": diff_data["absolute_difference"],
                            "dominant_demographic": seg1_name
                            if diff_data["difference_ratio"] > 0
                            else seg2_name,
                        })

        if major_rows:
            df = pd.DataFrame(major_rows)
            df.to_csv(
                self.output_dir / "academic_major_categories_comparison.csv",
                index=False,
            )
            print("Saved: academic_major_categories_comparison.csv")

    def _save_subcategories_comparison(self, comprehensive_comparisons: Dict):
        """Save subcategories comparison analysis."""

        sub_rows = []

        for group_id, comparison_data in comprehensive_comparisons.items():
            sub_diffs = comparison_data.get("subcategory_differences", {})

            for col, col_data in sub_diffs.items():
                notable_diffs = col_data.get("most_notable", [])

                for subcategory, diff_data in notable_diffs:
                    # Extract demographic names dynamically
                    segment_names = []
                    for key in diff_data.keys():
                        if key.endswith("_count"):
                            segment_names.append(key.replace("_count", ""))

                    if len(segment_names) == 2:
                        seg1_name, seg2_name = segment_names
                        sub_rows.append({
                            "demographic_group": group_id,
                            "subcategory": subcategory,
                            f"{seg1_name}_count": diff_data[f"{seg1_name}_count"],
                            f"{seg2_name}_count": diff_data[f"{seg2_name}_count"],
                            f"{seg1_name}_rate": round(
                                diff_data[f"{seg1_name}_rate"], 4
                            ),
                            f"{seg2_name}_rate": round(
                                diff_data[f"{seg2_name}_rate"], 4
                            ),
                            "difference_ratio": round(diff_data["difference_ratio"], 4),
                            "absolute_difference": diff_data["absolute_difference"],
                            "dominant_demographic": seg1_name
                            if diff_data["difference_ratio"] > 0
                            else seg2_name,
                        })

        if sub_rows:
            df = pd.DataFrame(sub_rows)
            df.to_csv(
                self.output_dir / "academic_subcategories_comparison.csv", index=False
            )
            print("Saved: academic_subcategories_comparison.csv")

    def _save_dominant_clauses_table(self, results: Dict):
        """Save dominant topic clauses as CSV."""

        clauses_data = results.get("dominant_topic_clauses", {})
        clause_rows = []

        for group_id, group_data in clauses_data.items():
            for segment_id, segment_data in group_data.items():
                for topic_key, topic_info in segment_data.items():
                    for i, clause in enumerate(topic_info.get("sample_clauses", [])):
                        clause_rows.append({
                            "demographic_group": group_id,
                            "segment": segment_id,
                            "topic": topic_info["topic"],
                            "topic_count": topic_info["count"],
                            "clause_number": i + 1,
                            "sample_clause": clause,
                        })

        if clause_rows:
            df = pd.DataFrame(clause_rows)
            df.to_csv(
                self.output_dir / "academic_dominant_topic_clauses.csv", index=False
            )
            print("Saved: academic_dominant_topic_clauses.csv")

    def _save_demographic_summary_table(self, results: Dict):
        """Save demographic analysis summary as CSV."""

        demographic_analyses = results.get("demographic_analyses", {})
        summary_rows = []

        for group_id, group_data in demographic_analyses.items():
            for segment_id, segment_data in group_data.get("segments", {}).items():
                segment_size = segment_data.get("segment_size", 0)
                segment_proportion = segment_data.get("segment_proportion", 0)

                for col, col_data in segment_data.items():
                    if isinstance(col_data, dict) and "top_topics" in col_data:
                        summary_rows.append({
                            "demographic_group": group_id,
                            "segment": segment_id,
                            "topic_column": col,
                            "segment_size": segment_size,
                            "segment_proportion": round(segment_proportion, 4),
                            "unique_topics": col_data.get("total_unique_topics", 0),
                            "topic_instances": col_data.get("total_topic_instances", 0),
                            "topic_coverage": round(
                                col_data.get("topic_coverage", 0), 4
                            ),
                            "avg_topics_per_record": round(
                                col_data.get("avg_topics_per_record", 0), 4
                            ),
                            "topic_concentration": round(
                                col_data.get("topic_concentration", 0), 4
                            ),
                            "dominant_topic": col_data["top_topics"][0][0]
                            if col_data["top_topics"]
                            else "",
                            "dominant_topic_count": col_data["top_topics"][0][1]
                            if col_data["top_topics"]
                            else 0,
                        })

        if summary_rows:
            df = pd.DataFrame(summary_rows)
            df.to_csv(self.output_dir / "academic_demographic_summary.csv", index=False)
            print("Saved: academic_demographic_summary.csv")

    def _create_group_summary(self, segments_with_data: List) -> Dict[str, Any]:
        """Create a summary for a demographic group."""

        return {
            "total_segments": len(segments_with_data),
            "total_records": sum(s["size"] for s in segments_with_data),
            "segment_sizes": {s["name"]: s["size"] for s in segments_with_data},
            "largest_segment": max(segments_with_data, key=lambda x: x["size"])["name"],
            "smallest_segment": min(segments_with_data, key=lambda x: x["size"])[
                "name"
            ],
        }


def test_academic_lda_analyzer():
    """Test the academic LDA demographic analyzer."""

    print("Testing Academic LDA Demographic Topic Analyzer...")

    # Load the sample data
    data_path = ".data/tables/lda_taxonomy_df.parquet"

    try:
        df = pl.read_parquet(data_path)
        print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

        # Initialize academic analyzer
        analyzer = LDADemographicTopicAnalyzer()

        # Run analysis
        results = analyzer.analyze_lda_demographics(
            dataframe=df,
            top_n_topics=10,
            extract_clauses=True,
        )

        print("Academic LDA Demographics Analysis completed successfully!")
        print(f"Results summary:")
        print(
            f"   - Demographic groups analyzed: {len(results['demographic_analyses'])}"
        )
        print(
            f"   - Topic columns processed: {len(results['metadata']['topic_columns_analyzed'])}"
        )
        print(f"   - Total records processed: {results['metadata']['total_records']}")
        print(
            f"   - Gender comparisons created: {len(results['comprehensive_comparisons'])}"
        )

        return results

    except Exception as e:
        print(f"Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the test
    test_results = test_academic_lda_analyzer()
