"""
Demographic Topic Analysis Module - Focused LDA Analysis

This module provides comprehensive analysis capabilities for extracting and analyzing
topics across different demographic segments with separate files for each demographic.
"""

import json
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

PLOTLY_AVAILABLE = True
warnings.filterwarnings("ignore")


class LDADemographicTopicAnalyzer:
    """
    A comprehensive analyzer for extracting demographic-based topic insights from LDA blog data.

    This class creates separate analysis files for each demographic group to avoid cluttered
    visualizations and provide clear, focused insights.
    """

    def __init__(self, output_dir: str = ".data/lda_analysis_results"):
        """
        Initialize the analyzer with output directory configuration.

        Args:
            output_dir: Directory to save analysis results and visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define demographic groups with descriptive naming
        self.demographic_groups = {
            "all_genders": {
                "name": "All Genders",
                "segments": {
                    "all_genders_all_ages": {
                        "name": "All Genders - All Ages",
                        "filter": lambda df: df,
                    },
                    "all_genders_male": {
                        "name": "All Genders - Male Only",
                        "filter": lambda df: df.filter(
                            pl.col("gender").str.to_lowercase() == "male"
                        ),
                    },
                    "all_genders_female": {
                        "name": "All Genders - Female Only",
                        "filter": lambda df: df.filter(
                            pl.col("gender").str.to_lowercase() == "female"
                        ),
                    },
                },
            },
            "age_over_20": {
                "name": "Age Over 20",
                "segments": {
                    "age_over_20_all_genders": {
                        "name": "Age Over 20 - All Genders",
                        "filter": lambda df: df.filter(
                            pl.col("age").cast(pl.Int32) > 20
                        ),
                    },
                    "age_over_20_male": {
                        "name": "Age Over 20 - Male Only",
                        "filter": lambda df: df.filter(
                            (pl.col("age").cast(pl.Int32) > 20)
                            & (pl.col("gender").str.to_lowercase() == "male")
                        ),
                    },
                    "age_over_20_female": {
                        "name": "Age Over 20 - Female Only",
                        "filter": lambda df: df.filter(
                            (pl.col("age").cast(pl.Int32) > 20)
                            & (pl.col("gender").str.to_lowercase() == "female")
                        ),
                    },
                },
            },
            "age_under_20": {
                "name": "Age Under 20",
                "segments": {
                    "age_under_20_all_genders": {
                        "name": "Age Under 20 - All Genders",
                        "filter": lambda df: df.filter(
                            pl.col("age").cast(pl.Int32) <= 20
                        ),
                    },
                    "age_under_20_male": {
                        "name": "Age Under 20 - Male Only",
                        "filter": lambda df: df.filter(
                            (pl.col("age").cast(pl.Int32) <= 20)
                            & (pl.col("gender").str.to_lowercase() == "male")
                        ),
                    },
                    "age_under_20_female": {
                        "name": "Age Under 20 - Female Only",
                        "filter": lambda df: df.filter(
                            (pl.col("age").cast(pl.Int32) <= 20)
                            & (pl.col("gender").str.to_lowercase() == "female")
                        ),
                    },
                },
            },
            "age_group_below_20": {
                "name": "Age Group Below 20",
                "segments": {
                    "age_group_below_20_all_genders": {
                        "name": "Age Group Below 20 - All Genders",
                        "filter": lambda df: df.filter(
                            pl.col("age").cast(pl.Int32) < 20
                        ),
                    },
                    "age_group_below_20_male": {
                        "name": "Age Group Below 20 - Male Only",
                        "filter": lambda df: df.filter(
                            (pl.col("age").cast(pl.Int32) < 20)
                            & (pl.col("gender").str.to_lowercase() == "male")
                        ),
                    },
                    "age_group_below_20_female": {
                        "name": "Age Group Below 20 - Female Only",
                        "filter": lambda df: df.filter(
                            (pl.col("age").cast(pl.Int32) < 20)
                            & (pl.col("gender").str.to_lowercase() == "female")
                        ),
                    },
                },
            },
            "age_group_20_to_30": {
                "name": "Age Group 20-30",
                "segments": {
                    "age_group_20_to_30_all_genders": {
                        "name": "Age Group 20-30 - All Genders",
                        "filter": lambda df: df.filter(
                            (pl.col("age").cast(pl.Int32) >= 20)
                            & (pl.col("age").cast(pl.Int32) < 30)
                        ),
                    },
                    "age_group_20_to_30_male": {
                        "name": "Age Group 20-30 - Male Only",
                        "filter": lambda df: df.filter(
                            (pl.col("age").cast(pl.Int32) >= 20)
                            & (pl.col("age").cast(pl.Int32) < 30)
                            & (pl.col("gender").str.to_lowercase() == "male")
                        ),
                    },
                    "age_group_20_to_30_female": {
                        "name": "Age Group 20-30 - Female Only",
                        "filter": lambda df: df.filter(
                            (pl.col("age").cast(pl.Int32) >= 20)
                            & (pl.col("age").cast(pl.Int32) < 30)
                            & (pl.col("gender").str.to_lowercase() == "female")
                        ),
                    },
                },
            },
            "age_group_30_to_40": {
                "name": "Age Group 30-40",
                "segments": {
                    "age_group_30_to_40_all_genders": {
                        "name": "Age Group 30-40 - All Genders",
                        "filter": lambda df: df.filter(
                            (pl.col("age").cast(pl.Int32) >= 30)
                            & (pl.col("age").cast(pl.Int32) < 40)
                        ),
                    },
                    "age_group_30_to_40_male": {
                        "name": "Age Group 30-40 - Male Only",
                        "filter": lambda df: df.filter(
                            (pl.col("age").cast(pl.Int32) >= 30)
                            & (pl.col("age").cast(pl.Int32) < 40)
                            & (pl.col("gender").str.to_lowercase() == "male")
                        ),
                    },
                    "age_group_30_to_40_female": {
                        "name": "Age Group 30-40 - Female Only",
                        "filter": lambda df: df.filter(
                            (pl.col("age").cast(pl.Int32) >= 30)
                            & (pl.col("age").cast(pl.Int32) < 40)
                            & (pl.col("gender").str.to_lowercase() == "female")
                        ),
                    },
                },
            },
            "age_group_40_plus": {
                "name": "Age Group 40+",
                "segments": {
                    "age_group_40_plus_all_genders": {
                        "name": "Age Group 40+ - All Genders",
                        "filter": lambda df: df.filter(
                            pl.col("age").cast(pl.Int32) >= 40
                        ),
                    },
                    "age_group_40_plus_male": {
                        "name": "Age Group 40+ - Male Only",
                        "filter": lambda df: df.filter(
                            (pl.col("age").cast(pl.Int32) >= 40)
                            & (pl.col("gender").str.to_lowercase() == "male")
                        ),
                    },
                    "age_group_40_plus_female": {
                        "name": "Age Group 40+ - Female Only",
                        "filter": lambda df: df.filter(
                            (pl.col("age").cast(pl.Int32) >= 40)
                            & (pl.col("gender").str.to_lowercase() == "female")
                        ),
                    },
                },
            },
            "occupation_students": {
                "name": "Occupation - Students",
                "segments": {
                    "occupation_students_all_genders": {
                        "name": "Students - All Genders",
                        "filter": lambda df: df.filter(
                            pl.col("industry")
                            .str.to_lowercase()
                            .str.contains("student")
                        ),
                    },
                    "occupation_students_male": {
                        "name": "Students - Male Only",
                        "filter": lambda df: df.filter(
                            pl.col("industry")
                            .str.to_lowercase()
                            .str.contains("student")
                            & (pl.col("gender").str.to_lowercase() == "male")
                        ),
                    },
                    "occupation_students_female": {
                        "name": "Students - Female Only",
                        "filter": lambda df: df.filter(
                            pl.col("industry")
                            .str.to_lowercase()
                            .str.contains("student")
                            & (pl.col("gender").str.to_lowercase() == "female")
                        ),
                    },
                },
            },
        }

    def analyze_lda_demographics(
        self,
        dataframe: pl.DataFrame,
        topic_columns: List[str],
        top_n_topics: int = 10,
    ) -> Dict[str, Any]:
        """
        Comprehensive LDA demographic topic analysis with separate files for each demographic.

        Args:
            dataframe: Input dataframe with blog data and extracted topics
            topic_columns: List of column names containing topic data to analyze
            top_n_topics: Number of top topics to extract per demographic

        Returns:
            Dictionary containing comprehensive analysis results
        """

        print("🔍 Starting LDA Demographic Topic Analysis with separate files...")

        if dataframe.is_empty():
            raise ValueError("Input dataframe is empty")

        results = {
            "metadata": {
                "total_records": dataframe.height,
                "analysis_timestamp": datetime.now().isoformat(),
                "topic_columns_analyzed": topic_columns,
                "demographic_groups": list(self.demographic_groups.keys()),
            },
            "demographic_analyses": {},
            "validation_summary": {},
        }

        # Create validation summary first
        print("🔍 Validating demographic segments...")
        results["validation_summary"] = self._create_validation_summary(dataframe)

        # Analyze each demographic group separately
        for group_id, group_config in self.demographic_groups.items():
            print(f"\n📊 Analyzing demographic group: {group_config['name']}")

            group_results = self._analyze_demographic_group(
                dataframe, group_id, group_config, topic_columns, top_n_topics
            )

            results["demographic_analyses"][group_id] = group_results

        print(
            f"\n✅ LDA Demographic Analysis complete! Results saved to: {self.output_dir}"
        )
        return results

    def _create_validation_summary(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Create validation summary for all demographic segments."""

        validation_data = []
        summary_data = []

        for group_id, group_config in self.demographic_groups.items():
            print(f"  📋 Validating {group_config['name']}")

            for segment_id, segment_config in group_config["segments"].items():
                try:
                    filtered_df = segment_config["filter"](df)
                    segment_size = filtered_df.height
                    percentage = (segment_size / df.height) * 100

                    validation_data.append({
                        "demographic_group": group_config["name"],
                        "segment_name": segment_config["name"],
                        "segment_id": segment_id,
                        "segment_size": segment_size,
                        "percentage": percentage,
                        "has_data": segment_size > 0,
                    })

                    print(
                        f"    ✓ {segment_config['name']}: {segment_size} records ({percentage:.1f}%)"
                    )

                except Exception as e:
                    validation_data.append({
                        "demographic_group": group_config["name"],
                        "segment_name": segment_config["name"],
                        "segment_id": segment_id,
                        "segment_size": 0,
                        "percentage": 0,
                        "has_data": False,
                        "error": str(e),
                    })
                    print(f"    ❌ {segment_config['name']}: ERROR - {str(e)}")

        # Save validation table
        validation_df = pd.DataFrame(validation_data)
        validation_df.to_csv(
            self.output_dir / "lda_demographics_validation_summary.csv", index=False
        )

        # Create group summary
        groups_with_data = {}
        for group_id, group_config in self.demographic_groups.items():
            group_segments = [
                v
                for v in validation_data
                if v["demographic_group"] == group_config["name"]
            ]
            segments_with_data = [s for s in group_segments if s["has_data"]]

            groups_with_data[group_id] = {
                "name": group_config["name"],
                "total_segments": len(group_segments),
                "segments_with_data": len(segments_with_data),
                "has_any_data": len(segments_with_data) > 0,
            }

        print(f"\n📊 Validation Summary:")
        print(f"{'Group':<25} {'Segments w/ Data':<15} {'Status':<10}")
        print("-" * 50)

        for group_id, group_info in groups_with_data.items():
            status = "✅" if group_info["has_any_data"] else "❌"
            print(
                f"{group_info['name']:<25} {group_info['segments_with_data']}/{group_info['total_segments']:<15} {status:<10}"
            )

        return {"validation_data": validation_data, "groups_summary": groups_with_data}

    def _analyze_demographic_group(
        self,
        df: pl.DataFrame,
        group_id: str,
        group_config: Dict,
        topic_columns: List[str],
        top_n: int,
    ) -> Dict[str, Any]:
        """Analyze a single demographic group and create its dedicated files."""

        group_results = {
            "group_name": group_config["name"],
            "segments": {},
            "group_summary": {},
        }

        # Analyze each segment in this group
        segments_with_data = []

        for segment_id, segment_config in group_config["segments"].items():
            try:
                filtered_df = segment_config["filter"](df)
                segment_size = filtered_df.height

                if segment_size == 0:
                    print(f"    ⚠️  No data for {segment_config['name']}")
                    continue

                print(
                    f"    📈 Analyzing {segment_config['name']} ({segment_size} records)"
                )

                # Analyze topics for this segment
                segment_analysis = self._analyze_segment_topics(
                    filtered_df, segment_config["name"], topic_columns, top_n
                )
                segment_analysis["segment_size"] = segment_size
                segment_analysis["segment_percentage"] = (
                    segment_size / df.height
                ) * 100

                group_results["segments"][segment_id] = segment_analysis
                segments_with_data.append({
                    "id": segment_id,
                    "name": segment_config["name"],
                    "size": segment_size,
                    "analysis": segment_analysis,
                })

            except Exception as e:
                print(f"    ❌ Error analyzing {segment_config['name']}: {str(e)}")
                continue

        if segments_with_data:
            # Create dedicated visualizations for this group
            self._create_group_visualizations(
                group_id, group_config, segments_with_data, topic_columns
            )

            # Create dedicated analysis tables for this group
            self._create_group_tables(
                group_id, group_config, segments_with_data, topic_columns
            )

            # Create group summary
            group_results["group_summary"] = self._create_group_summary(
                segments_with_data
            )

        return group_results

    def _analyze_segment_topics(
        self, df: pl.DataFrame, segment_name: str, topic_columns: List[str], top_n: int
    ) -> Dict[str, Any]:
        """Analyze topics for a single demographic segment."""

        segment_results = {}

        for col in topic_columns:
            # Extract and aggregate topics
            topic_counter = Counter()
            major_category_counter = Counter()
            subcategory_counter = Counter()

            for row in df.select(col).to_pandas()[col]:
                if pd.isna(row) or row is None:
                    continue

                try:
                    # Handle different data formats
                    if isinstance(row, str):
                        if row.startswith("{") or row.startswith("["):
                            topics_data = json.loads(row)
                        else:
                            topics_data = [topic.strip() for topic in row.split(",")]
                    else:
                        topics_data = row

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

                                        weight = max(1, int(round(score)))
                                        topic_counter[taxonomy_key] += weight
                                        major_category_counter[major] += weight
                                        subcategory_counter[subcategory] += weight
                                else:
                                    topic_counter[taxonomy_key] += 1
                        else:
                            for key, value in topics_data.items():
                                if isinstance(value, (list, tuple)):
                                    topic_counter.update(value)
                                else:
                                    topic_counter[key] += 1
                    elif isinstance(topics_data, (list, tuple)):
                        if col == "lda_taxonomy_classification":
                            for item in topics_data:
                                if isinstance(item, str) and ":" in item:
                                    parts = item.split(":", 1)
                                    if len(parts) == 2:
                                        major, subcategory = parts
                                        major = major.strip()
                                        subcategory = subcategory.strip()

                                        topic_counter[item] += 1
                                        major_category_counter[major] += 1
                                        subcategory_counter[subcategory] += 1
                                else:
                                    topic_counter[str(item)] += 1
                        else:
                            topic_counter.update(topics_data)
                    elif isinstance(topics_data, str):
                        topic_counter[topics_data] += 1

                except (json.JSONDecodeError, TypeError, AttributeError):
                    continue

            # Get top topics
            top_topics = topic_counter.most_common(top_n)
            total_topics = sum(topic_counter.values())

            result = {
                "top_topics": top_topics,
                "total_unique_topics": len(topic_counter),
                "total_topic_instances": total_topics,
                "topic_diversity": len(topic_counter) / total_topics
                if total_topics > 0
                else 0,
                "full_distribution": dict(topic_counter),
            }

            # Add taxonomy-specific analysis
            if col == "lda_taxonomy_classification" and (
                major_category_counter or subcategory_counter
            ):
                result["taxonomy_analysis"] = {
                    "major_categories": {
                        "top_categories": major_category_counter.most_common(top_n),
                        "total_unique_categories": len(major_category_counter),
                        "total_category_instances": sum(
                            major_category_counter.values()
                        ),
                        "full_distribution": dict(major_category_counter),
                    },
                    "subcategories": {
                        "top_subcategories": subcategory_counter.most_common(top_n),
                        "total_unique_subcategories": len(subcategory_counter),
                        "total_subcategory_instances": sum(
                            subcategory_counter.values()
                        ),
                        "full_distribution": dict(subcategory_counter),
                    },
                }

            segment_results[col] = result

        return segment_results

    def _create_group_visualizations(
        self,
        group_id: str,
        group_config: Dict,
        segments_with_data: List,
        topic_columns: List[str],
    ):
        """Create dedicated visualization files for a demographic group."""

        print(f"    🎨 Creating visualizations for {group_config['name']}")

        # Create charts for each topic column
        for col in topic_columns:
            self._create_group_topic_charts(
                group_id, group_config, segments_with_data, col
            )

            # Create taxonomy charts if applicable
            if col == "lda_taxonomy_classification":
                self._create_group_taxonomy_charts(
                    group_id, group_config, segments_with_data, col
                )

    def _create_group_topic_charts(
        self,
        group_id: str,
        group_config: Dict,
        segments_with_data: List,
        column_name: str,
    ):
        """Create topic distribution charts for a demographic group."""

        # Set up the plot style
        try:
            plt.style.use("seaborn-v0_8")
        except OSError:
            try:
                plt.style.use("seaborn")
            except OSError:
                # Use default style if seaborn is not available
                plt.style.use("default")

        sns.set_palette("husl")

        # Calculate number of segments for subplot layout
        n_segments = len(segments_with_data)

        # Fix subplot configuration based on number of segments
        if n_segments == 1:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            axes = [ax]
        elif n_segments == 2:
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            if not isinstance(axes, np.ndarray):
                axes = [axes]
        elif n_segments == 3:
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        elif n_segments == 4:
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            axes = axes.flatten()
        else:
            n_cols = 3
            n_rows = (n_segments + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 8 * n_rows))
            if n_rows == 1:
                axes = [axes] if n_segments == 1 else axes
            else:
                axes = axes.flatten()

        fig.suptitle(
            f"LDA Topic Analysis: {group_config['name']} - {column_name}",
            fontsize=16,
            fontweight="bold",
        )

        for i, segment_data in enumerate(segments_with_data):
            if i >= len(axes):
                break

            ax = axes[i]
            segment_analysis = segment_data["analysis"]

            if column_name not in segment_analysis:
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{segment_data['name']}\nNo Data")
                continue

            topic_data = segment_analysis[column_name]
            top_topics = topic_data["top_topics"][:8]  # Limit to top 8 for readability

            if not top_topics:
                ax.text(
                    0.5,
                    0.5,
                    "No topics found",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{segment_data['name']}\nNo Topics")
                continue

            # Create horizontal bar chart
            topics, counts = zip(*top_topics)
            y_pos = np.arange(len(topics))

            # Truncate long topic names for better display
            display_topics = [
                topic[:40] + "..." if len(topic) > 40 else topic for topic in topics
            ]

            bars = ax.barh(
                y_pos,
                counts,
                color=cm.get_cmap("viridis")(np.linspace(0, 1, len(topics))),
            )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(display_topics, fontsize=9)
            ax.set_xlabel("Frequency", fontsize=10)
            ax.set_title(
                f"{segment_data['name']}\n({segment_data['size']} records)", fontsize=11
            )
            ax.grid(axis="x", alpha=0.3)

            # Add value labels
            for bar, count in zip(bars, counts):
                ax.text(
                    bar.get_width() + 0.01 * max(counts),
                    bar.get_y() + bar.get_height() / 2,
                    str(count),
                    va="center",
                    fontsize=9,
                )

        # Hide unused subplots
        for i in range(len(segments_with_data), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # Save with descriptive filename
        filename = f"lda_demographics_{group_id}_{column_name}_topic_analysis.png"
        filepath = self.output_dir / filename

        try:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"      📊 Saved: {filename}")
        except Exception as e:
            print(f"      ❌ Failed to save {filename}: {str(e)}")
        finally:
            plt.close()

    def _create_group_taxonomy_charts(
        self,
        group_id: str,
        group_config: Dict,
        segments_with_data: List,
        column_name: str,
    ):
        """Create taxonomy-specific charts for a demographic group."""

        # Create major categories chart
        self._create_taxonomy_category_chart(
            group_id,
            group_config,
            segments_with_data,
            column_name,
            "major_categories",
            "Major Categories",
        )

        # Create subcategories chart
        self._create_taxonomy_category_chart(
            group_id,
            group_config,
            segments_with_data,
            column_name,
            "subcategories",
            "Subcategories",
        )

    def _create_taxonomy_category_chart(
        self,
        group_id: str,
        group_config: Dict,
        segments_with_data: List,
        column_name: str,
        category_type: str,
        category_label: str,
    ):
        """Create a chart for taxonomy categories (major or sub)."""

        try:
            plt.style.use("seaborn-v0_8")
        except OSError:
            try:
                plt.style.use("seaborn")
            except OSError:
                plt.style.use("default")

        n_segments = len(segments_with_data)

        # Fix subplot configuration based on number of segments
        if n_segments == 1:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            axes = [ax]
        elif n_segments == 2:
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            if not isinstance(axes, np.ndarray):
                axes = [axes]
        elif n_segments == 3:
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        elif n_segments == 4:
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            axes = axes.flatten()
        else:
            n_cols = 3
            n_rows = (n_segments + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 8 * n_rows))
            if n_rows == 1:
                axes = [axes] if n_segments == 1 else axes
            else:
                axes = axes.flatten()

        fig.suptitle(
            f"LDA {category_label} Analysis: {group_config['name']}",
            fontsize=16,
            fontweight="bold",
        )

        for i, segment_data in enumerate(segments_with_data):
            if i >= len(axes):
                break

            ax = axes[i]
            segment_analysis = segment_data["analysis"]

            if (
                column_name not in segment_analysis
                or "taxonomy_analysis" not in segment_analysis[column_name]
                or category_type
                not in segment_analysis[column_name]["taxonomy_analysis"]
            ):
                ax.text(
                    0.5,
                    0.5,
                    "No taxonomy data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{segment_data['name']}\nNo Data")
                continue

            taxonomy_data = segment_analysis[column_name]["taxonomy_analysis"][
                category_type
            ]

            if category_type == "major_categories":
                top_items = taxonomy_data["top_categories"][:8]
            else:
                top_items = taxonomy_data["top_subcategories"][:8]

            if not top_items:
                ax.text(
                    0.5,
                    0.5,
                    f"No {category_label.lower()}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{segment_data['name']}\nNo {category_label}")
                continue

            # Create horizontal bar chart
            categories, counts = zip(*top_items)
            y_pos = np.arange(len(categories))

            # Truncate long category names
            display_categories = [
                cat[:30] + "..." if len(cat) > 30 else cat for cat in categories
            ]

            bars = ax.barh(
                y_pos,
                counts,
                color=cm.get_cmap("plasma")(np.linspace(0, 1, len(categories))),
            )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(display_categories, fontsize=9)
            ax.set_xlabel("Frequency", fontsize=10)
            ax.set_title(
                f"{segment_data['name']}\n({segment_data['size']} records)", fontsize=11
            )
            ax.grid(axis="x", alpha=0.3)

            # Add value labels
            for bar, count in zip(bars, counts):
                ax.text(
                    bar.get_width() + 0.01 * max(counts),
                    bar.get_y() + bar.get_height() / 2,
                    str(count),
                    va="center",
                    fontsize=9,
                )

        # Hide unused subplots
        for i in range(len(segments_with_data), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # Save with descriptive filename
        filename = f"lda_demographics_{group_id}_{category_type}_analysis.png"
        filepath = self.output_dir / filename

        try:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"      📊 Saved: {filename}")
        except Exception as e:
            print(f"      ❌ Failed to save {filename}: {str(e)}")
        finally:
            plt.close()

    def _create_group_tables(
        self,
        group_id: str,
        group_config: Dict,
        segments_with_data: List,
        topic_columns: List[str],
    ):
        """Create dedicated analysis tables for a demographic group."""

        print(f"    📋 Creating tables for {group_config['name']}")

        # Create main topic analysis table for this group
        group_topic_data = []

        for segment_data in segments_with_data:
            segment_analysis = segment_data["analysis"]

            for col in topic_columns:
                if col in segment_analysis and "top_topics" in segment_analysis[col]:
                    for topic, count in segment_analysis[col]["top_topics"]:
                        group_topic_data.append({
                            "demographic_group": group_config["name"],
                            "segment_name": segment_data["name"],
                            "segment_id": segment_data["id"],
                            "column": col,
                            "topic": topic,
                            "frequency": count,
                            "segment_size": segment_data["size"],
                            "topic_diversity": segment_analysis[col].get(
                                "topic_diversity", 0
                            ),
                        })

        if group_topic_data:
            filename = f"lda_demographics_{group_id}_topic_analysis.csv"
            filepath = self.output_dir / filename
            try:
                pd.DataFrame(group_topic_data).to_csv(filepath, index=False)
                print(f"      📊 Saved: {filename}")
            except Exception as e:
                print(f"      ❌ Failed to save {filename}: {str(e)}")

        # Create taxonomy tables if applicable
        for col in topic_columns:
            if col == "lda_taxonomy_classification":
                self._create_group_taxonomy_tables(
                    group_id, group_config, segments_with_data, col
                )

    def _create_group_taxonomy_tables(
        self,
        group_id: str,
        group_config: Dict,
        segments_with_data: List,
        column_name: str,
    ):
        """Create taxonomy-specific tables for a demographic group."""

        # Major categories table
        major_cat_data = []
        subcategory_data = []

        for segment_data in segments_with_data:
            segment_analysis = segment_data["analysis"]

            if (
                column_name in segment_analysis
                and "taxonomy_analysis" in segment_analysis[column_name]
            ):
                taxonomy_analysis = segment_analysis[column_name]["taxonomy_analysis"]

                # Major categories
                if "major_categories" in taxonomy_analysis:
                    for category, count in taxonomy_analysis["major_categories"][
                        "top_categories"
                    ]:
                        major_cat_data.append({
                            "demographic_group": group_config["name"],
                            "segment_name": segment_data["name"],
                            "segment_id": segment_data["id"],
                            "major_category": category,
                            "frequency": count,
                            "segment_size": segment_data["size"],
                        })

                # Subcategories
                if "subcategories" in taxonomy_analysis:
                    for subcategory, count in taxonomy_analysis["subcategories"][
                        "top_subcategories"
                    ]:
                        subcategory_data.append({
                            "demographic_group": group_config["name"],
                            "segment_name": segment_data["name"],
                            "segment_id": segment_data["id"],
                            "subcategory": subcategory,
                            "frequency": count,
                            "segment_size": segment_data["size"],
                        })

        # Save tables
        if major_cat_data:
            filename = f"lda_demographics_{group_id}_major_categories.csv"
            filepath = self.output_dir / filename
            try:
                pd.DataFrame(major_cat_data).to_csv(filepath, index=False)
                print(f"      📊 Saved: {filename}")
            except Exception as e:
                print(f"      ❌ Failed to save {filename}: {str(e)}")

        if subcategory_data:
            filename = f"lda_demographics_{group_id}_subcategories.csv"
            filepath = self.output_dir / filename
            try:
                pd.DataFrame(subcategory_data).to_csv(filepath, index=False)
                print(f"      📊 Saved: {filename}")
            except Exception as e:
                print(f"      ❌ Failed to save {filename}: {str(e)}")

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


def test_lda_demographic_analyzer():
    """Test the new LDA demographic analyzer."""

    print("🧪 Testing new LDA Demographic Topic Analyzer...")

    # Load the sample data
    data_path = (
        "/Users/t834527/Repos/comp814-assignment2/.data/tables/lda_taxonomy_df.parquet"
    )

    try:
        df = pl.read_parquet(data_path)
        print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

        # Initialize new analyzer
        analyzer = LDADemographicTopicAnalyzer()

        # Define topic columns to analyze
        topic_columns = ["lda_topics", "lda_taxonomy_classification"]

        # Run analysis
        results = analyzer.analyze_lda_demographics(
            dataframe=df,
            topic_columns=topic_columns,
            top_n_topics=10,
        )

        print("\n🎉 New LDA Demographics Analysis completed successfully!")
        print(f"📊 Results summary:")
        print(
            f"   - Demographic groups analyzed: {len(results['demographic_analyses'])}"
        )
        print(f"   - Topic columns processed: {len(topic_columns)}")
        print(f"   - Total records processed: {results['metadata']['total_records']}")

        return results

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the test
    test_results = test_lda_demographic_analyzer()
