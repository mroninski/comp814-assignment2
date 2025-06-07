import polars as pl
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


class TopicTaxonomyResultsAggregator:
    """
    A class to aggregate topic taxonomy classifications by demographics for the assignment.

    This class processes topic taxonomy results and aggregates them across different
    demographic segments including gender, age groups, industry (students), and overall population.
    Classifications are weighted by their probability scores and normalized by the number of
    documents in each demographic group to provide fair comparisons and more accurate topic
    distributions. The class supports both category-only and category+sub-category aggregations.
    """

    def __init__(
        self, parquet_file_path: str, column_name: str = "lda_taxonomy_classification"
    ):
        """
        Initialize the aggregator with data from a parquet file.

        Args:
            parquet_file_path (str): Path to the parquet file containing LDA results
        """
        self.df = pl.read_parquet(parquet_file_path)
        self.column_name = column_name
        self.category_aggregations = {}
        self.category_subcategory_aggregations = {}
        self._prepare_demographics()
        self._aggregate_by_demographics()

    def _prepare_demographics(self):
        """Prepare demographic groupings based on assignment requirements."""
        # Create age groups: <=20 and >20
        self.df = self.df.with_columns([
            pl.col("age").cast(pl.Float64, strict=False).alias("age_numeric"),
        ]).with_columns([
            pl.when(pl.col("age_numeric").is_not_null() & (pl.col("age_numeric") <= 20))
            .then(pl.lit("<=20"))
            .otherwise(pl.lit(">20"))
            .alias("age_group")
        ])

        # Create student indicator
        self.df = self.df.with_columns([
            (pl.col("industry") == "Student").alias("is_student")
        ])

    def _parse_lda_classification_json(self, json_str: str) -> Dict[str, float]:
        """
        Parse the LDA taxonomy classification JSON string.

        Args:
            json_str (str): JSON string containing classification probabilities

        Returns:
            Dict[str, float]: Dictionary of classifications and their probabilities
        """
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return {}

    def _extract_category_from_classification(self, classification: str) -> str:
        """
        Extract the category part from a 'category:sub-category' classification.

        Args:
            classification (str): Full classification in format 'category:sub-category'

        Returns:
            str: Category part only
        """
        return (
            classification.split(":", 1)[0] if ":" in classification else classification
        )

    def _aggregate_category_classifications_for_group(
        self, group_df: pl.DataFrame
    ) -> Dict[str, float]:
        """
        Aggregate and normalize weighted classifications for a demographic group, using category only.
        The normalization ensures that scores are comparable across demographic groups of different sizes
        by calculating the average topic probability per document.

        Args:
            group_df (pl.DataFrame): DataFrame subset for the demographic group

        Returns:
            Dict[str, float]: Normalized weighted classification scores by category.
                             Each score represents the average topic probability per document in the group.
        """
        aggregated_scores = defaultdict(float)
        num_docs = group_df.height

        if num_docs == 0:
            return {}

        for row in group_df.iter_rows(named=True):
            classifications = self._parse_lda_classification_json(row[self.column_name])

            # Add weighted scores for each classification, using category only
            for classification, probability in classifications.items():
                category = self._extract_category_from_classification(classification)
                aggregated_scores[category] += (
                    probability / 100.0
                )  # Convert percentage to weight

        # Normalize scores by the number of documents in the group
        normalized_scores = {
            category: total_score / num_docs
            for category, total_score in aggregated_scores.items()
        }

        return normalized_scores

    def _aggregate_category_subcategory_classifications_for_group(
        self, group_df: pl.DataFrame
    ) -> Dict[str, float]:
        """
        Aggregate and normalize weighted classifications for a demographic group, using full category:sub-category.
        The normalization ensures that scores are comparable across demographic groups of different sizes
        by calculating the average topic probability per document.

        Args:
            group_df (pl.DataFrame): DataFrame subset for the demographic group

        Returns:
            Dict[str, float]: Normalized weighted classification scores by category+sub-category.
                             Each score represents the average topic probability per document in the group.
        """
        aggregated_scores = defaultdict(float)
        num_docs = group_df.height

        if num_docs == 0:
            return {}

        for row in group_df.iter_rows(named=True):
            classifications = self._parse_lda_classification_json(row[self.column_name])

            # Add weighted scores for each classification
            for classification, probability in classifications.items():
                aggregated_scores[classification] += (
                    probability / 100.0
                )  # Convert percentage to weight

        # Normalize scores by the number of documents in the group
        normalized_scores = {
            classification: total_score / num_docs
            for classification, total_score in aggregated_scores.items()
        }

        return normalized_scores

    def _aggregate_by_demographics(self) -> None:
        """
        Aggregate classifications across all demographic groups for both category-only and category+sub-category.
        """
        demographics = {
            "Male": self.df.filter(pl.col("gender") == "male"),
            "Female": self.df.filter(pl.col("gender") == "female"),
            "Age <=20": self.df.filter(pl.col("age_group") == "<=20"),
            "Age >20": self.df.filter(pl.col("age_group") == ">20"),
            "Students": self.df.filter(pl.col("is_student") == True),  # noqa: E712
            "Non-Students": self.df.filter(pl.col("is_student") == False),  # noqa: E712
            "Everyone": self.df,
        }

        category_results = {}
        category_subcategory_results = {}

        for demo_name, demo_df in demographics.items():
            if demo_df.height > 0:  # Use height instead of empty for polars
                category_results[demo_name] = (
                    self._aggregate_category_classifications_for_group(demo_df)
                )
                category_subcategory_results[demo_name] = (
                    self._aggregate_category_subcategory_classifications_for_group(
                        demo_df
                    )
                )

        self.category_aggregations = category_results
        self.category_subcategory_aggregations = category_subcategory_results

    def get_top_classifications(
        self, classifications: Dict[str, float], top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top N classifications sorted by normalized weighted score.

        Args:
            classifications (Dict[str, float]): Classification scores
            top_n (int): Number of top classifications to return

        Returns:
            List[Tuple[str, float]]: List of (classification, score) tuples
        """
        return sorted(classifications.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def validate_filename(self, filename: str | Path):
        """
        Validate the filename.
        """
        if isinstance(filename, str):
            filename = Path(filename)

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        return filename

    def save_category_demographics_to_parquet(self, filename: str | Path):
        """
        Save the category-only demographic aggregations to a parquet file.

        Args:
            filename (str): Output filename for the parquet file
        """
        filename = self.validate_filename(filename)

        # Convert aggregations to a structured format suitable for parquet
        records = []
        for demographic, classifications in self.category_aggregations.items():
            for position, (classification, average_weighted_score) in enumerate(
                classifications.items()
            ):
                records.append({
                    "demographic": demographic,
                    "classification": classification,
                    "average_weighted_score": average_weighted_score,
                })

        # Create DataFrame and save to parquet
        df_results = pl.DataFrame(records)
        ranked_df_results = self.creat_and_sort_rank_column(df_results)
        ranked_df_results.write_parquet(filename)
        ranked_df_results.write_csv(filename.with_suffix(".csv"))
        print(f"Category demographic aggregations saved to {filename}")

    def save_category_subcategory_demographics_to_parquet(self, filename: str | Path):
        """
        Save the category+sub-category demographic aggregations to a parquet file.

        Args:
            filename (str): Output filename for the parquet file
        """
        filename = self.validate_filename(filename)

        # Convert aggregations to a structured format suitable for parquet
        records = []
        for (
            demographic,
            classifications,
        ) in self.category_subcategory_aggregations.items():
            for classification, average_weighted_score in classifications.items():
                records.append({
                    "demographic": demographic,
                    "classification": classification,
                    "average_weighted_score": average_weighted_score,
                })

        # Create DataFrame and save to parquet
        df_results = pl.DataFrame(records)

        ranked_df_results = self.creat_and_sort_rank_column(df_results)

        ranked_df_results.write_parquet(filename)
        ranked_df_results.write_csv(filename.with_suffix(".csv"))
        print(f"Category+Sub-category demographic aggregations saved to {filename}")

    def creat_and_sort_rank_column(self, temp_df: pl.DataFrame) -> pl.DataFrame:
        """
        Create and sort a rank column based on the average_weighted_score within each demographic.
        """
        # The rank should be based on the average_weighted_score within each demographic
        df_results = temp_df.with_columns(
            pl.col("average_weighted_score")
            .rank(method="ordinal", descending=True)
            .over("demographic")
            .alias("rank")
        )

        # Just to make the output more readable, we sort by demographic
        df_results = df_results.sort(["demographic", "rank"])

        return df_results

    def get_category_demographics_dataframe(self) -> pl.DataFrame:
        """
        Get the category-only demographic aggregations as a polars DataFrame.

        Returns:
            pl.DataFrame: DataFrame with columns [demographic, classification, average_weighted_score]
        """
        records = []
        for demographic, classifications in self.category_aggregations.items():
            for classification, average_weighted_score in classifications.items():
                records.append({
                    "demographic": demographic,
                    "classification": classification,
                    "average_weighted_score": average_weighted_score,
                })

        return pl.DataFrame(records)

    def get_category_subcategory_demographics_dataframe(self) -> pl.DataFrame:
        """
        Get the category+sub-category demographic aggregations as a polars DataFrame.

        Returns:
            pl.DataFrame: DataFrame with columns [demographic, classification, average_weighted_score]
        """
        records = []
        for (
            demographic,
            classifications,
        ) in self.category_subcategory_aggregations.items():
            for classification, average_weighted_score in classifications.items():
                records.append({
                    "demographic": demographic,
                    "classification": classification,
                    "average_weighted_score": average_weighted_score,
                })

        return pl.DataFrame(records)

    def get_category_demographic_summary(self) -> Dict[str, Tuple[str, str]]:
        """
        Get a summary of the top 2 categories for each demographic.

        Returns:
            Dict[str, Tuple[str, str]]: Dictionary mapping demographics to their top 2 categories
        """
        summary = {}
        for demo_name, classifications in self.category_aggregations.items():
            top_2 = self.get_top_classifications(classifications, 2)
            if len(top_2) >= 2:
                summary[demo_name] = (top_2[0][0], top_2[1][0])
            elif len(top_2) == 1:
                summary[demo_name] = (top_2[0][0], "No second category")
            else:
                summary[demo_name] = ("No categories found", "No categories found")

        return summary

    def get_category_subcategory_demographic_summary(
        self,
    ) -> Dict[str, Tuple[str, str]]:
        """
        Get a summary of the top 2 category+sub-category topics for each demographic.

        Returns:
            Dict[str, Tuple[str, str]]: Dictionary mapping demographics to their top 2 topics
        """
        summary = {}
        for (
            demo_name,
            classifications,
        ) in self.category_subcategory_aggregations.items():
            top_2 = self.get_top_classifications(classifications, 2)
            if len(top_2) >= 2:
                summary[demo_name] = (top_2[0][0], top_2[1][0])
            elif len(top_2) == 1:
                summary[demo_name] = (top_2[0][0], "No second topic")
            else:
                summary[demo_name] = ("No topics found", "No topics found")

        return summary
