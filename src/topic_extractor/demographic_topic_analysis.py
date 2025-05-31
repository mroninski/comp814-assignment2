"""
Demographic Topic Analysis Module

This module provides comprehensive analysis capabilities for extracting and analyzing
topics across different demographic segments.
"""

import json
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import seaborn as sns
from plotly.subplots import make_subplots

PLOTLY_AVAILABLE = True


warnings.filterwarnings("ignore")


class DemographicTopicAnalyzer:
    """
    A comprehensive analyzer for extracting demographic-based topic insights from blog data.

    This class implements multiple analysis strategies including topic extraction, demographic segmentation, temporal analysis, and visualization.
    """

    def __init__(self, output_dir: str = ".data/analysis_results"):
        """
        Initialize the analyzer with output directory configuration.

        Args:
            output_dir: Directory to save analysis results and visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize demographic mappings based on assignment requirements
        self.demographic_configs = {
            "gender": {
                "male": lambda df: df.filter(
                    pl.col("gender").str.to_lowercase() == "male"
                ),
                "female": lambda df: df.filter(
                    pl.col("gender").str.to_lowercase() == "female"
                ),
                "everyone": lambda df: df,
            },
            "age_groups": {
                "age_20_and_under": lambda df: df.filter(
                    pl.col("age").cast(pl.Int32) <= 20
                ),
                "age_over_20": lambda df: df.filter(pl.col("age").cast(pl.Int32) > 20),
                "everyone": lambda df: df,
            },
            "occupation": {
                "students": lambda df: df.filter(
                    pl.col("industry").str.to_lowercase().str.contains("student")
                ),
                "everyone": lambda df: df,
            },
        }

    def analyze_demographic_topics(
        self,
        dataframe: pl.DataFrame,
        topic_columns: List[str],
        custom_demographics: Optional[Dict[str, Dict[str, Any]]] = None,
        top_n_topics: int = 10,
        date_range_analysis: bool = True,
    ) -> Dict[str, Any]:
        """
        Comprehensive demographic topic analysis as specified in the COMP814 assignment.

        This function performs the core analysis required by the assignment, extracting
        the most popular topics across different demographic segments.

        Args:
            dataframe: Input dataframe with blog data and extracted topics
            topic_columns: List of column names containing topic data to analyze
            custom_demographics: Additional demographic configurations beyond the default
            top_n_topics: Number of top topics to extract per demographic
            date_range_analysis: Whether to perform temporal topic analysis

        Returns:
            Dictionary containing comprehensive analysis results including:
            - Overall topic frequencies
            - Demographic-specific topic rankings
            - Temporal analysis results
            - Statistical summaries
        """

        print("🔍 Starting comprehensive demographic topic analysis...")

        # Ensure we have valid data
        if dataframe.is_empty():
            raise ValueError("Input dataframe is empty")

        # Merge custom demographics if provided
        demographics = self.demographic_configs.copy()
        if custom_demographics:
            demographics.update(custom_demographics)

        results = {
            "metadata": {
                "total_records": dataframe.height,
                "analysis_timestamp": datetime.now().isoformat(),
                "topic_columns_analyzed": topic_columns,
                "demographics_analyzed": list(demographics.keys()),
            },
            "overall_analysis": {},
            "demographic_analysis": {},
            "temporal_analysis": {},
            "comparative_analysis": {},
            "statistical_summary": {},
        }

        # 1. Overall Topic Analysis
        print("📊 Analyzing overall topic distribution...")
        results["overall_analysis"] = self._analyze_overall_topics(
            dataframe, topic_columns, top_n_topics
        )

        # 2. Demographic-Specific Analysis
        print("👥 Analyzing topics by demographic segments...")
        results["demographic_analysis"] = self._analyze_demographic_segments(
            dataframe, topic_columns, demographics, top_n_topics
        )

        # 3. Temporal Analysis (if date column exists and requested)
        if date_range_analysis and "date" in dataframe.columns:
            print("📅 Performing temporal topic analysis...")
            results["temporal_analysis"] = self._analyze_temporal_topics(
                dataframe, topic_columns, top_n_topics
            )

        # 4. Comparative Analysis
        print("🔄 Conducting comparative demographic analysis...")
        results["comparative_analysis"] = self._perform_comparative_analysis(
            results["demographic_analysis"], topic_columns
        )

        # 5. Statistical Summary
        print("📈 Generating statistical summaries...")
        results["statistical_summary"] = self._generate_statistical_summary(
            dataframe, results, topic_columns
        )

        # 6. Generate Visualizations
        print("🎨 Creating comprehensive visualizations...")
        self._generate_visualizations(results, topic_columns)

        # 7. Save Analysis Tables
        print("💾 Saving detailed analysis tables...")
        self._save_analysis_tables(results, topic_columns)

        print("✅ Analysis complete! Results saved to:", self.output_dir)
        return results

    def _analyze_overall_topics(
        self, df: pl.DataFrame, topic_columns: List[str], top_n: int
    ) -> Dict[str, Any]:
        """Analyze overall topic distribution across the entire dataset."""
        overall_results = {}

        for col in topic_columns:
            print(f"  📋 Processing column: {col}")

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
                            # JSON format
                            topics_data = json.loads(row)
                        else:
                            # Simple string format
                            topics_data = [topic.strip() for topic in row.split(",")]
                    else:
                        topics_data = row

                    # Extract topics based on data structure
                    if isinstance(topics_data, dict):
                        # Special handling for lda_taxonomy_classification
                        if col == "lda_taxonomy_classification":
                            for taxonomy_key, score in topics_data.items():
                                if (
                                    isinstance(taxonomy_key, str)
                                    and ":" in taxonomy_key
                                ):
                                    # Split major:subcategory format
                                    parts = taxonomy_key.split(
                                        ":", 1
                                    )  # Split only on first ":"
                                    if len(parts) == 2:
                                        major, subcategory = parts
                                        major = major.strip()
                                        subcategory = subcategory.strip()

                                        # Use score as weight for counting (convert to int for frequency)
                                        weight = max(
                                            1, int(round(score))
                                        )  # Ensure at least 1

                                        # Count full taxonomy item
                                        topic_counter[taxonomy_key] += weight
                                        # Count major category separately
                                        major_category_counter[major] += weight
                                        # Count subcategory separately
                                        subcategory_counter[subcategory] += weight
                                else:
                                    # Fallback for items without ":" format
                                    topic_counter[taxonomy_key] += 1
                        else:
                            # Handle regular dictionary format (taxonomy classifications or nested structures)
                            for key, value in topics_data.items():
                                if isinstance(value, (list, tuple)):
                                    topic_counter.update(value)
                                else:
                                    topic_counter[key] += 1
                    elif isinstance(topics_data, (list, tuple)):
                        # Special handling for lda_taxonomy_classification if it were a list
                        if col == "lda_taxonomy_classification":
                            for item in topics_data:
                                if isinstance(item, str) and ":" in item:
                                    # Split major:subcategory format
                                    parts = item.split(
                                        ":", 1
                                    )  # Split only on first ":"
                                    if len(parts) == 2:
                                        major, subcategory = parts
                                        major = major.strip()
                                        subcategory = subcategory.strip()

                                        # Count full taxonomy item
                                        topic_counter[item] += 1
                                        # Count major category separately
                                        major_category_counter[major] += 1
                                        # Count subcategory separately
                                        subcategory_counter[subcategory] += 1
                                else:
                                    # Fallback for items without ":" format
                                    topic_counter[str(item)] += 1
                        else:
                            # Regular list handling for other columns
                            topic_counter.update(topics_data)
                    elif isinstance(topics_data, str):
                        topic_counter[topics_data] += 1

                except (json.JSONDecodeError, TypeError, AttributeError) as e:
                    # Handle malformed data gracefully
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

            # Add taxonomy-specific analysis for lda_taxonomy_classification
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

            overall_results[col] = result

        return overall_results

    def _analyze_demographic_segments(
        self,
        df: pl.DataFrame,
        topic_columns: List[str],
        demographics: Dict[str, Dict[str, Any]],
        top_n: int,
    ) -> Dict[str, Any]:
        """Analyze topics for each demographic segment."""
        demographic_results = {}

        for demo_category, demo_filters in demographics.items():
            print(f"  👥 Analyzing demographic category: {demo_category}")
            demographic_results[demo_category] = {}

            for demo_name, filter_func in demo_filters.items():
                print(f"    🔍 Processing segment: {demo_name}")

                # Apply demographic filter
                try:
                    filtered_df = filter_func(df)
                    segment_size = filtered_df.height

                    if segment_size == 0:
                        print(f"      ⚠️  No data found for segment: {demo_name}")
                        continue

                    # Analyze topics for this demographic segment
                    segment_results = self._analyze_overall_topics(
                        filtered_df, topic_columns, top_n
                    )
                    segment_results["segment_size"] = segment_size
                    segment_results["segment_percentage"] = (
                        segment_size / df.height
                    ) * 100

                    demographic_results[demo_category][demo_name] = segment_results

                except Exception as e:
                    print(f"      ❌ Error processing segment {demo_name}: {str(e)}")
                    continue

        return demographic_results

    def _analyze_temporal_topics(
        self, df: pl.DataFrame, topic_columns: List[str], top_n: int
    ) -> Dict[str, Any]:
        """Analyze topic trends over time periods."""
        temporal_results = {}

        # Convert date column to datetime for temporal analysis
        try:
            # Handle different date formats
            if df["date"].dtype == pl.Float64:
                # Assume Unix timestamp
                temporal_df = df.with_columns(
                    pl.from_epoch(pl.col("date").cast(pl.Int64), time_unit="s").alias(
                        "parsed_date"
                    )
                )
            else:
                temporal_df = df.with_columns(
                    pl.col("date")
                    .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
                    .alias("parsed_date")
                )
        except Exception as e:
            print(f"    ⚠️  Date parsing failed: {str(e)}")
            return temporal_results

        # Define time periods for analysis
        time_periods = {
            "yearly": lambda df: df.with_columns(
                pl.col("parsed_date").dt.year().alias("period")
            ),
            "monthly": lambda df: df.with_columns(
                (
                    pl.col("parsed_date").dt.year().cast(pl.Utf8)
                    + "-"
                    + pl.col("parsed_date").dt.month().cast(pl.Utf8).str.zfill(2)
                ).alias("period")
            ),
            "quarterly": lambda df: df.with_columns(
                (
                    pl.col("parsed_date").dt.year().cast(pl.Utf8)
                    + "-Q"
                    + ((pl.col("parsed_date").dt.month() - 1) // 3 + 1).cast(pl.Utf8)
                ).alias("period")
            ),
        }

        for period_name, period_func in time_periods.items():
            print(f"    📅 Analyzing {period_name} trends...")

            try:
                period_df = period_func(temporal_df)
                periods = (
                    period_df.select("period").unique().to_pandas()["period"].tolist()
                )

                period_results = {}
                for period in periods:
                    period_data = period_df.filter(pl.col("period") == period)
                    if period_data.height > 0:
                        period_analysis = self._analyze_overall_topics(
                            period_data, topic_columns, top_n
                        )
                        period_results[str(period)] = period_analysis

                temporal_results[period_name] = period_results

            except Exception as e:
                print(f"      ❌ Error in {period_name} analysis: {str(e)}")
                continue

        return temporal_results

    def _perform_comparative_analysis(
        self, demographic_results: Dict[str, Any], topic_columns: List[str]
    ) -> Dict[str, Any]:
        """Perform comparative analysis across demographic segments."""
        comparative_results = {}

        for col in topic_columns:
            comparative_results[col] = {
                "demographic_topic_overlap": {},
                "unique_demographic_topics": {},
                "topic_concentration_analysis": {},
            }

            # Collect all topics by demographic
            demo_topics = {}
            for demo_category, segments in demographic_results.items():
                for segment_name, segment_data in segments.items():
                    if col in segment_data:
                        demo_key = f"{demo_category}_{segment_name}"
                        demo_topics[demo_key] = set(
                            topic
                            for topic, _ in segment_data[col].get("top_topics", [])
                        )

            # Calculate overlaps and unique topics
            demo_keys = list(demo_topics.keys())
            for i, demo1 in enumerate(demo_keys):
                for demo2 in demo_keys[i + 1 :]:
                    if demo1 in demo_topics and demo2 in demo_topics:
                        overlap = demo_topics[demo1] & demo_topics[demo2]
                        unique_1 = demo_topics[demo1] - demo_topics[demo2]
                        unique_2 = demo_topics[demo2] - demo_topics[demo1]

                        comparative_results[col]["demographic_topic_overlap"][
                            f"{demo1}_vs_{demo2}"
                        ] = {
                            "overlap_topics": list(overlap),
                            "overlap_count": len(overlap),
                            f"{demo1}_unique": list(unique_1),
                            f"{demo2}_unique": list(unique_2),
                            "jaccard_similarity": len(overlap)
                            / len(demo_topics[demo1] | demo_topics[demo2])
                            if demo_topics[demo1] | demo_topics[demo2]
                            else 0,
                        }

        return comparative_results

    def _generate_statistical_summary(
        self, df: pl.DataFrame, results: Dict[str, Any], topic_columns: List[str]
    ) -> Dict[str, Any]:
        """Generate comprehensive statistical summaries."""
        stats = {
            "dataset_statistics": {
                "total_records": df.height,
                "total_columns": len(df.columns),
                "data_completeness": {},
            },
            "demographic_statistics": {},
            "topic_statistics": {},
        }

        # Data completeness analysis
        for col in df.columns:
            non_null_count = df.select(pl.col(col).is_not_null().sum()).item()
            stats["dataset_statistics"]["data_completeness"][col] = {
                "non_null_count": non_null_count,
                "completeness_percentage": (non_null_count / df.height) * 100,
            }

        # Demographic distribution statistics
        for demo_col in ["gender", "age", "industry"]:
            if demo_col in df.columns:
                distribution = (
                    df.select(demo_col).to_pandas()[demo_col].value_counts().to_dict()
                )
                stats["demographic_statistics"][demo_col] = {
                    "distribution": distribution,
                    "unique_values": len(distribution),
                    "most_common": max(distribution.items(), key=lambda x: x[1])
                    if distribution
                    else None,
                }

        # Topic analysis statistics
        for col in topic_columns:
            if col in results["overall_analysis"]:
                col_stats = results["overall_analysis"][col]
                stats["topic_statistics"][col] = {
                    "total_unique_topics": col_stats["total_unique_topics"],
                    "total_topic_instances": col_stats["total_topic_instances"],
                    "topic_diversity_score": col_stats["topic_diversity"],
                    "top_topic_dominance": col_stats["top_topics"][0][1]
                    / col_stats["total_topic_instances"]
                    if col_stats["top_topics"]
                    else 0,
                }

        return stats

    def _generate_visualizations(
        self, results: Dict[str, Any], topic_columns: List[str]
    ) -> None:
        """Generate comprehensive visualizations for the analysis results."""
        print("    🎨 Creating topic distribution charts...")

        # Set style for better visualizations
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # 1. Overall Topic Distribution Visualization
        for col in topic_columns:
            if col in results["overall_analysis"]:
                self._create_topic_distribution_charts(
                    col, results["overall_analysis"][col]
                )

        # 2. Demographic Comparison Visualizations
        self._create_demographic_comparison_charts(
            results["demographic_analysis"], topic_columns
        )

        # 3. Temporal Analysis Visualizations
        if results["temporal_analysis"]:
            self._create_temporal_analysis_charts(
                results["temporal_analysis"], topic_columns
            )

        # 4. Interactive Dashboard-style Visualization
        self._create_interactive_dashboard(results, topic_columns)

    def _create_topic_distribution_charts(
        self, column_name: str, topic_data: Dict[str, Any]
    ) -> None:
        """Create distribution charts for individual topic columns."""
        top_topics = topic_data["top_topics"][:10]  # Top 10 for visibility

        if not top_topics:
            return

        # Check if this is taxonomy data with special analysis
        has_taxonomy = "taxonomy_analysis" in topic_data

        if has_taxonomy:
            # Create extended figure for taxonomy analysis
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
                3, 2, figsize=(20, 22)
            )
        else:
            # Regular figure layout
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

        fig.suptitle(
            f"Topic Analysis for {column_name}", fontsize=16, fontweight="bold"
        )

        # 1. Horizontal bar chart for full topics
        topics, counts = zip(*top_topics)
        y_pos = np.arange(len(topics))

        bars = ax1.barh(
            y_pos, counts, color=cm.get_cmap("viridis")(np.linspace(0, 1, len(topics)))
        )
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([t[:30] + "..." if len(t) > 30 else t for t in topics])
        ax1.set_xlabel("Frequency")
        ax1.set_title("Top Topics Distribution")
        ax1.grid(axis="x", alpha=0.3)

        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax1.text(
                bar.get_width() + 0.01 * max(counts),
                bar.get_y() + bar.get_height() / 2,
                str(count),
                va="center",
                fontsize=9,
            )

        # 2. Pie chart for top 8 topics
        top_8_topics = top_topics[:8]
        if len(top_8_topics) < len(top_topics):
            other_count = sum(count for _, count in top_topics[8:])
            pie_data = list(top_8_topics) + [("Others", other_count)]
        else:
            pie_data = top_8_topics

        labels, sizes = zip(*pie_data)
        colors = cm.get_cmap("Set3")(np.linspace(0, 1, len(labels)))

        wedges, texts, autotexts = ax2.pie(
            sizes,
            labels=[l[:20] + "..." if len(l) > 20 else l for l in labels],
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        ax2.set_title("Topic Distribution (Percentage)")

        # 3. Cumulative distribution
        sorted_counts = sorted(topic_data["full_distribution"].values(), reverse=True)
        cumulative_pct = np.cumsum(sorted_counts) / sum(sorted_counts) * 100

        ax3.plot(
            range(1, len(cumulative_pct) + 1),
            cumulative_pct,
            "bo-",
            linewidth=2,
            markersize=3,
        )
        ax3.set_xlabel("Topic Rank")
        ax3.set_ylabel("Cumulative Percentage")
        ax3.set_title("Cumulative Topic Distribution")
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=80, color="r", linestyle="--", alpha=0.7, label="80% threshold")
        ax3.legend()

        # 4. Topic frequency histogram
        frequencies = list(topic_data["full_distribution"].values())
        ax4.hist(
            frequencies,
            bins=min(30, len(set(frequencies))),
            color="skyblue",
            alpha=0.7,
            edgecolor="black",
        )
        ax4.set_xlabel("Topic Frequency")
        ax4.set_ylabel("Number of Topics")
        ax4.set_title("Topic Frequency Distribution")
        ax4.grid(axis="y", alpha=0.3)

        # 5. & 6. Taxonomy-specific charts (if available)
        if has_taxonomy:
            taxonomy_data = topic_data["taxonomy_analysis"]

            # 5. Major categories bar chart
            if "major_categories" in taxonomy_data:
                major_cats = taxonomy_data["major_categories"]["top_categories"][:10]
                if major_cats:
                    cat_names, cat_counts = zip(*major_cats)
                    y_pos_cat = np.arange(len(cat_names))

                    bars = ax5.barh(
                        y_pos_cat,
                        cat_counts,
                        color=cm.get_cmap("plasma")(np.linspace(0, 1, len(cat_names))),
                    )
                    ax5.set_yticks(y_pos_cat)
                    ax5.set_yticklabels([
                        c[:25] + "..." if len(c) > 25 else c for c in cat_names
                    ])
                    ax5.set_xlabel("Frequency")
                    ax5.set_title("Top Major Categories")
                    ax5.grid(axis="x", alpha=0.3)

                    # Add value labels
                    for bar, count in zip(bars, cat_counts):
                        ax5.text(
                            bar.get_width() + 0.01 * max(cat_counts),
                            bar.get_y() + bar.get_height() / 2,
                            str(count),
                            va="center",
                            fontsize=9,
                        )

            # 6. Subcategories bar chart
            if "subcategories" in taxonomy_data:
                subcats = taxonomy_data["subcategories"]["top_subcategories"][:10]
                if subcats:
                    sub_names, sub_counts = zip(*subcats)
                    y_pos_sub = np.arange(len(sub_names))

                    bars = ax6.barh(
                        y_pos_sub,
                        sub_counts,
                        color=cm.get_cmap("magma")(np.linspace(0, 1, len(sub_names))),
                    )
                    ax6.set_yticks(y_pos_sub)
                    ax6.set_yticklabels([
                        s[:25] + "..." if len(s) > 25 else s for s in sub_names
                    ])
                    ax6.set_xlabel("Frequency")
                    ax6.set_title("Top Subcategories")
                    ax6.grid(axis="x", alpha=0.3)

                    # Add value labels
                    for bar, count in zip(bars, sub_counts):
                        ax6.text(
                            bar.get_width() + 0.01 * max(sub_counts),
                            bar.get_y() + bar.get_height() / 2,
                            str(count),
                            va="center",
                            fontsize=9,
                        )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"topic_distribution_{column_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_demographic_comparison_charts(
        self, demographic_data: Dict[str, Any], topic_columns: List[str]
    ) -> None:
        """Create comparison charts across demographic segments."""
        for col in topic_columns:
            # Collect demographic data for comparison
            demo_comparison = {}

            for demo_category, segments in demographic_data.items():
                for segment_name, segment_data in segments.items():
                    if col in segment_data and "top_topics" in segment_data[col]:
                        demo_key = f"{demo_category}_{segment_name}"
                        demo_comparison[demo_key] = {
                            "topics": dict(
                                segment_data[col]["top_topics"][:5]
                            ),  # Top 5 for comparison
                            "size": segment_data.get("segment_size", 0),
                            "diversity": segment_data[col].get("topic_diversity", 0),
                        }

            if len(demo_comparison) < 2:
                continue

            # Create comparison visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
            fig.suptitle(
                f"Demographic Comparison for {col}", fontsize=16, fontweight="bold"
            )

            # 1. Segment sizes
            segments = list(demo_comparison.keys())
            sizes = [demo_comparison[seg]["size"] for seg in segments]

            bars = ax1.bar(
                range(len(segments)),
                sizes,
                color=cm.get_cmap("tab10")(np.linspace(0, 1, len(segments))),
            )
            ax1.set_xticks(range(len(segments)))
            ax1.set_xticklabels(
                [s.replace("_", "\n") for s in segments], rotation=45, ha="right"
            )
            ax1.set_ylabel("Segment Size")
            ax1.set_title("Demographic Segment Sizes")
            ax1.grid(axis="y", alpha=0.3)

            # Add value labels on bars
            for bar, size in zip(bars, sizes):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(sizes),
                    str(size),
                    ha="center",
                    va="bottom",
                )

            # 2. Topic diversity comparison
            diversities = [demo_comparison[seg]["diversity"] for seg in segments]

            bars = ax2.bar(
                range(len(segments)),
                diversities,
                color=cm.get_cmap("viridis")(np.linspace(0, 1, len(segments))),
            )
            ax2.set_xticks(range(len(segments)))
            ax2.set_xticklabels(
                [s.replace("_", "\n") for s in segments], rotation=45, ha="right"
            )
            ax2.set_ylabel("Topic Diversity Score")
            ax2.set_title("Topic Diversity by Demographic")
            ax2.grid(axis="y", alpha=0.3)

            # 3. Top topics heatmap
            all_topics = set()
            for seg_data in demo_comparison.values():
                all_topics.update(seg_data["topics"].keys())

            all_topics = list(all_topics)[:15]  # Limit to top 15 for visibility

            heatmap_data = []
            for segment in segments:
                row = []
                for topic in all_topics:
                    row.append(demo_comparison[segment]["topics"].get(topic, 0))
                heatmap_data.append(row)

            im = ax3.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")
            ax3.set_xticks(range(len(all_topics)))
            ax3.set_xticklabels(
                [t[:15] + "..." if len(t) > 15 else t for t in all_topics],
                rotation=45,
                ha="right",
            )
            ax3.set_yticks(range(len(segments)))
            ax3.set_yticklabels([s.replace("_", "\n") for s in segments])
            ax3.set_title("Topic Frequency Heatmap")

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax3)
            cbar.set_label("Topic Frequency")

            # 4. Topic overlap analysis
            if len(segments) >= 2:
                overlap_matrix = np.zeros((len(segments), len(segments)))

                for i, seg1 in enumerate(segments):
                    for j, seg2 in enumerate(segments):
                        if i != j:
                            topics1 = set(demo_comparison[seg1]["topics"].keys())
                            topics2 = set(demo_comparison[seg2]["topics"].keys())

                            if topics1 or topics2:
                                jaccard = len(topics1 & topics2) / len(
                                    topics1 | topics2
                                )
                                overlap_matrix[i][j] = jaccard

                im = ax4.imshow(overlap_matrix, cmap="Blues", vmin=0, vmax=1)
                ax4.set_xticks(range(len(segments)))
                ax4.set_xticklabels(
                    [s.replace("_", "\n") for s in segments], rotation=45, ha="right"
                )
                ax4.set_yticks(range(len(segments)))
                ax4.set_yticklabels([s.replace("_", "\n") for s in segments])
                ax4.set_title("Topic Overlap (Jaccard Similarity)")

                # Add text annotations
                for i in range(len(segments)):
                    for j in range(len(segments)):
                        text = ax4.text(
                            j,
                            i,
                            f"{overlap_matrix[i, j]:.2f}",
                            ha="center",
                            va="center",
                            color="white" if overlap_matrix[i, j] > 0.5 else "black",
                        )

                plt.colorbar(im, ax=ax4)

            plt.tight_layout()
            plt.savefig(
                self.output_dir / f"demographic_comparison_{col}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def _create_temporal_analysis_charts(
        self, temporal_data: Dict[str, Any], topic_columns: List[str]
    ) -> None:
        """Create temporal analysis visualizations."""
        for period_type, period_data in temporal_data.items():
            if not period_data:
                continue

            for col in topic_columns:
                # Extract temporal topic trends
                periods = sorted(period_data.keys())

                # Get top topics across all periods
                all_topics = Counter()
                for period in periods:
                    if (
                        col in period_data[period]
                        and "top_topics" in period_data[period][col]
                    ):
                        for topic, count in period_data[period][col]["top_topics"]:
                            all_topics[topic] += count

                top_temporal_topics = [topic for topic, _ in all_topics.most_common(10)]

                # Create temporal trend chart
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
                fig.suptitle(
                    f"Temporal Analysis - {col} ({period_type})",
                    fontsize=16,
                    fontweight="bold",
                )

                # 1. Topic trends over time
                for i, topic in enumerate(top_temporal_topics[:5]):  # Top 5 for clarity
                    topic_counts = []
                    for period in periods:
                        if (
                            col in period_data[period]
                            and "full_distribution" in period_data[period][col]
                        ):
                            count = period_data[period][col]["full_distribution"].get(
                                topic, 0
                            )
                        else:
                            count = 0
                        topic_counts.append(count)

                    ax1.plot(
                        periods,
                        topic_counts,
                        marker="o",
                        linewidth=2,
                        label=topic[:25] + "..." if len(topic) > 25 else topic,
                    )

                ax1.set_xlabel("Time Period")
                ax1.set_ylabel("Topic Frequency")
                ax1.set_title("Topic Trends Over Time")
                ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis="x", rotation=45)

                # 2. Topic diversity over time
                diversity_scores = []
                total_topics = []

                for period in periods:
                    if (
                        col in period_data[period]
                        and "topic_diversity" in period_data[period][col]
                    ):
                        diversity_scores.append(
                            period_data[period][col]["topic_diversity"]
                        )
                        total_topics.append(
                            period_data[period][col]["total_unique_topics"]
                        )
                    else:
                        diversity_scores.append(0)
                        total_topics.append(0)

                ax2_twin = ax2.twinx()

                line1 = ax2.plot(
                    periods,
                    diversity_scores,
                    "b-o",
                    linewidth=2,
                    label="Diversity Score",
                )
                line2 = ax2_twin.plot(
                    periods,
                    total_topics,
                    "r-s",
                    linewidth=2,
                    label="Total Unique Topics",
                )

                ax2.set_xlabel("Time Period")
                ax2.set_ylabel("Topic Diversity Score", color="b")
                ax2_twin.set_ylabel("Total Unique Topics", color="r")
                ax2.set_title("Topic Diversity Evolution")
                ax2.tick_params(axis="x", rotation=45)
                ax2.grid(True, alpha=0.3)

                # Combine legends
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax2.legend(lines, labels, loc="upper left")

                plt.tight_layout()
                plt.savefig(
                    self.output_dir / f"temporal_analysis_{period_type}_{col}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

    def _create_interactive_dashboard(
        self, results: Dict[str, Any], topic_columns: List[str]
    ) -> None:
        """Create an interactive dashboard using Plotly."""
        if not PLOTLY_AVAILABLE:
            print("    ⚠️  Plotly not available for interactive dashboard")
            return

        try:
            import plotly.offline as pyo

            # Create subplots for dashboard
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Overall Topic Distribution",
                    "Demographic Comparison",
                    "Topic Diversity Analysis",
                    "Temporal Trends",
                ),
                specs=[
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"type": "scatter"}, {"type": "scatter"}],
                ],
            )

            # Add overall topic distribution
            for i, col in enumerate(
                topic_columns[:1]
            ):  # First column only for simplicity
                if col in results["overall_analysis"]:
                    topics_data = results["overall_analysis"][col]["top_topics"][:10]
                    if topics_data:
                        topics, counts = zip(*topics_data)
                        fig.add_trace(
                            go.Bar(
                                x=list(topics), y=list(counts), name=f"{col} Topics"
                            ),
                            row=1,
                            col=1,
                        )

            # Add demographic comparison
            demo_segments = []
            demo_sizes = []
            for demo_category, segments in results["demographic_analysis"].items():
                for segment_name, segment_data in segments.items():
                    demo_segments.append(f"{demo_category}_{segment_name}")
                    demo_sizes.append(segment_data.get("segment_size", 0))

            if demo_segments:
                fig.add_trace(
                    go.Bar(x=demo_segments, y=demo_sizes, name="Segment Sizes"),
                    row=1,
                    col=2,
                )

            # Update layout
            fig.update_layout(
                height=800,
                title_text="Comprehensive Topic Analysis Dashboard",
                showlegend=True,
                title_x=0.5,
            )

            # Save interactive dashboard
            pyo.plot(
                fig,
                filename=str(self.output_dir / "interactive_dashboard.html"),
                auto_open=False,
            )

        except ImportError:
            print("    ⚠️  Plotly not available for interactive dashboard")

    def _save_analysis_tables(
        self, results: Dict[str, Any], topic_columns: List[str]
    ) -> None:
        """Save comprehensive analysis tables to CSV files."""

        # 1. Overall topic analysis table
        overall_tables = []
        for col in topic_columns:
            if col in results["overall_analysis"]:
                topic_data = results["overall_analysis"][col]
                for topic, count in topic_data["top_topics"]:
                    overall_tables.append({
                        "column": col,
                        "topic": topic,
                        "frequency": count,
                        "percentage": (count / topic_data["total_topic_instances"])
                        * 100
                        if topic_data["total_topic_instances"] > 0
                        else 0,
                    })

        if overall_tables:
            pd.DataFrame(overall_tables).to_csv(
                self.output_dir / "overall_topic_analysis.csv", index=False
            )

        # 1a. Taxonomy-specific analysis tables
        taxonomy_major_tables = []
        taxonomy_sub_tables = []

        for col in topic_columns:
            if (
                col in results["overall_analysis"]
                and "taxonomy_analysis" in results["overall_analysis"][col]
            ):
                taxonomy_data = results["overall_analysis"][col]["taxonomy_analysis"]

                # Major categories table
                if "major_categories" in taxonomy_data:
                    major_data = taxonomy_data["major_categories"]
                    for category, count in major_data["top_categories"]:
                        taxonomy_major_tables.append({
                            "column": col,
                            "major_category": category,
                            "frequency": count,
                            "percentage": (
                                count / major_data["total_category_instances"]
                            )
                            * 100
                            if major_data["total_category_instances"] > 0
                            else 0,
                        })

                # Subcategories table
                if "subcategories" in taxonomy_data:
                    sub_data = taxonomy_data["subcategories"]
                    for subcategory, count in sub_data["top_subcategories"]:
                        taxonomy_sub_tables.append({
                            "column": col,
                            "subcategory": subcategory,
                            "frequency": count,
                            "percentage": (
                                count / sub_data["total_subcategory_instances"]
                            )
                            * 100
                            if sub_data["total_subcategory_instances"] > 0
                            else 0,
                        })

        if taxonomy_major_tables:
            pd.DataFrame(taxonomy_major_tables).to_csv(
                self.output_dir / "taxonomy_major_categories.csv", index=False
            )

        if taxonomy_sub_tables:
            pd.DataFrame(taxonomy_sub_tables).to_csv(
                self.output_dir / "taxonomy_subcategories.csv", index=False
            )

        # 2. Demographic analysis tables
        demographic_tables = []
        for demo_category, segments in results["demographic_analysis"].items():
            for segment_name, segment_data in segments.items():
                for col in topic_columns:
                    if col in segment_data and "top_topics" in segment_data[col]:
                        for topic, count in segment_data[col]["top_topics"]:
                            demographic_tables.append({
                                "demographic_category": demo_category,
                                "demographic_segment": segment_name,
                                "column": col,
                                "topic": topic,
                                "frequency": count,
                                "segment_size": segment_data.get("segment_size", 0),
                                "topic_diversity": segment_data[col].get(
                                    "topic_diversity", 0
                                ),
                            })

        if demographic_tables:
            pd.DataFrame(demographic_tables).to_csv(
                self.output_dir / "demographic_topic_analysis.csv", index=False
            )

        # 2a. Demographic taxonomy analysis tables
        demo_taxonomy_major_tables = []
        demo_taxonomy_sub_tables = []

        for demo_category, segments in results["demographic_analysis"].items():
            for segment_name, segment_data in segments.items():
                for col in topic_columns:
                    if col in segment_data and "taxonomy_analysis" in segment_data[col]:
                        taxonomy_data = segment_data[col]["taxonomy_analysis"]

                        # Major categories for this demographic segment
                        if "major_categories" in taxonomy_data:
                            major_data = taxonomy_data["major_categories"]
                            for category, count in major_data["top_categories"]:
                                demo_taxonomy_major_tables.append({
                                    "demographic_category": demo_category,
                                    "demographic_segment": segment_name,
                                    "column": col,
                                    "major_category": category,
                                    "frequency": count,
                                    "segment_size": segment_data.get("segment_size", 0),
                                })

                        # Subcategories for this demographic segment
                        if "subcategories" in taxonomy_data:
                            sub_data = taxonomy_data["subcategories"]
                            for subcategory, count in sub_data["top_subcategories"]:
                                demo_taxonomy_sub_tables.append({
                                    "demographic_category": demo_category,
                                    "demographic_segment": segment_name,
                                    "column": col,
                                    "subcategory": subcategory,
                                    "frequency": count,
                                    "segment_size": segment_data.get("segment_size", 0),
                                })

        if demo_taxonomy_major_tables:
            pd.DataFrame(demo_taxonomy_major_tables).to_csv(
                self.output_dir / "demographic_taxonomy_major_categories.csv",
                index=False,
            )

        if demo_taxonomy_sub_tables:
            pd.DataFrame(demo_taxonomy_sub_tables).to_csv(
                self.output_dir / "demographic_taxonomy_subcategories.csv", index=False
            )

        # 3. Temporal analysis tables
        if results["temporal_analysis"]:
            temporal_tables = []
            for period_type, period_data in results["temporal_analysis"].items():
                for period, period_analysis in period_data.items():
                    for col in topic_columns:
                        if (
                            col in period_analysis
                            and "top_topics" in period_analysis[col]
                        ):
                            for topic, count in period_analysis[col]["top_topics"]:
                                temporal_tables.append({
                                    "period_type": period_type,
                                    "period": period,
                                    "column": col,
                                    "topic": topic,
                                    "frequency": count,
                                    "period_diversity": period_analysis[col].get(
                                        "topic_diversity", 0
                                    ),
                                })

            if temporal_tables:
                pd.DataFrame(temporal_tables).to_csv(
                    self.output_dir / "temporal_topic_analysis.csv", index=False
                )

        # 4. Statistical summary table
        stats_table = []
        for stat_category, stat_data in results["statistical_summary"].items():
            if isinstance(stat_data, dict):
                for key, value in stat_data.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            stats_table.append({
                                "category": stat_category,
                                "metric": key,
                                "sub_metric": sub_key,
                                "value": sub_value,
                            })
                    else:
                        stats_table.append({
                            "category": stat_category,
                            "metric": key,
                            "sub_metric": "",
                            "value": value,
                        })

        if stats_table:
            pd.DataFrame(stats_table).to_csv(
                self.output_dir / "statistical_summary.csv", index=False
            )

        # 5. Comparative analysis table
        if results["comparative_analysis"]:
            comparison_tables = []
            for col, col_data in results["comparative_analysis"].items():
                for comparison_type, comparisons in col_data.items():
                    if isinstance(comparisons, dict):
                        for comparison_name, comparison_data in comparisons.items():
                            if isinstance(comparison_data, dict):
                                comparison_tables.append({
                                    "column": col,
                                    "comparison_type": comparison_type,
                                    "comparison": comparison_name,
                                    "overlap_count": comparison_data.get(
                                        "overlap_count", 0
                                    ),
                                    "jaccard_similarity": comparison_data.get(
                                        "jaccard_similarity", 0
                                    ),
                                    "overlap_topics": ";".join(
                                        comparison_data.get("overlap_topics", [])
                                    ),
                                })

            if comparison_tables:
                pd.DataFrame(comparison_tables).to_csv(
                    self.output_dir / "comparative_analysis.csv", index=False
                )

        print(f"    💾 Analysis tables saved to {self.output_dir}")
        print(
            f"    📊 Additional taxonomy analysis tables created for major categories and subcategories"
        )


def test_analyzer_with_sample_data():
    """
    Test function to validate the analyzer with the provided sample data.

    This function serves as both a test and an example of how to use the
    DemographicTopicAnalyzer class.
    """
    print("🧪 Testing DemographicTopicAnalyzer with sample data...")

    # Load the sample data
    data_path = (
        "/Users/t834527/Repos/comp814-assignment2/.data/tables/lda_taxonomy_df.parquet"
    )

    try:
        df = pl.read_parquet(data_path)
        print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

        # Initialize analyzer
        analyzer = DemographicTopicAnalyzer(output_dir=".data/test_analysis_results")

        # Define topic columns to analyze
        topic_columns = ["lda_topic_words", "lda_taxonomy_classification"]

        # Run comprehensive analysis
        results = analyzer.analyze_demographic_topics(
            dataframe=df,
            topic_columns=topic_columns,
            top_n_topics=5,  # Smaller number for test data
            date_range_analysis=True,
        )

        print("🎉 Analysis completed successfully!")
        print(f"📊 Results summary:")
        print(f"   - Demographics analyzed: {len(results['demographic_analysis'])}")
        print(f"   - Topic columns processed: {len(topic_columns)}")
        print(f"   - Total records processed: {results['metadata']['total_records']}")

        return results

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the test
    test_results = test_analyzer_with_sample_data()
