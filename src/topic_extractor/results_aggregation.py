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
    Classifications are weighted by their probability scores to provide more accurate topic
    distributions. The class supports both category-only and category+sub-category aggregations.
    """

    def __init__(self, parquet_file_path: str):
        """
        Initialize the aggregator with data from a parquet file.

        Args:
            parquet_file_path (str): Path to the parquet file containing LDA results
        """
        self.df = pl.read_parquet(parquet_file_path)
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
        Aggregate weighted classifications for a demographic group, using category only.

        Args:
            group_df (pl.DataFrame): DataFrame subset for the demographic group

        Returns:
            Dict[str, float]: Aggregated weighted classification scores by category
        """
        aggregated_scores = defaultdict(float)

        for row in group_df.iter_rows(named=True):
            classifications = self._parse_lda_classification_json(
                row["lda_taxonomy_classification"]
            )

            # Add weighted scores for each classification, using category only
            for classification, probability in classifications.items():
                category = self._extract_category_from_classification(classification)
                aggregated_scores[category] += (
                    probability / 100.0
                )  # Convert percentage to weight

        return dict(aggregated_scores)

    def _aggregate_category_subcategory_classifications_for_group(
        self, group_df: pl.DataFrame
    ) -> Dict[str, float]:
        """
        Aggregate weighted classifications for a demographic group, using full category:sub-category.

        Args:
            group_df (pl.DataFrame): DataFrame subset for the demographic group

        Returns:
            Dict[str, float]: Aggregated weighted classification scores by category+sub-category
        """
        aggregated_scores = defaultdict(float)

        for row in group_df.iter_rows(named=True):
            classifications = self._parse_lda_classification_json(
                row["lda_taxonomy_classification"]
            )

            # Add weighted scores for each classification
            for classification, probability in classifications.items():
                aggregated_scores[classification] += (
                    probability / 100.0
                )  # Convert percentage to weight

        return dict(aggregated_scores)

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
        Get top N classifications sorted by weighted score.

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
            for position, (classification, weighted_score) in enumerate(
                classifications.items()
            ):
                records.append({
                    "demographic": demographic,
                    "position": position + 1,
                    "classification": classification,
                    "weighted_score": weighted_score,
                })

        # Create DataFrame and save to parquet
        df_results = pl.DataFrame(records)
        df_results.write_parquet(filename)
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
            for position, (classification, weighted_score) in enumerate(
                classifications.items()
            ):
                records.append({
                    "demographic": demographic,
                    "position": position + 1,
                    "classification": classification,
                    "weighted_score": weighted_score,
                })

        # Create DataFrame and save to parquet
        df_results = pl.DataFrame(records)
        df_results.write_parquet(filename)
        print(f"Category+Sub-category demographic aggregations saved to {filename}")

    def get_category_demographics_dataframe(self) -> pl.DataFrame:
        """
        Get the category-only demographic aggregations as a polars DataFrame.

        Returns:
            pl.DataFrame: DataFrame with columns [demographic, classification, weighted_score]
        """
        records = []
        for demographic, classifications in self.category_aggregations.items():
            for classification, weighted_score in classifications.items():
                records.append({
                    "demographic": demographic,
                    "classification": classification,
                    "weighted_score": weighted_score,
                })

        return pl.DataFrame(records)

    def get_category_subcategory_demographics_dataframe(self) -> pl.DataFrame:
        """
        Get the category+sub-category demographic aggregations as a polars DataFrame.

        Returns:
            pl.DataFrame: DataFrame with columns [demographic, classification, weighted_score]
        """
        records = []
        for (
            demographic,
            classifications,
        ) in self.category_subcategory_aggregations.items():
            for classification, weighted_score in classifications.items():
                records.append({
                    "demographic": demographic,
                    "classification": classification,
                    "weighted_score": weighted_score,
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

    def save_category_report(self, filename: str | Path, top_n: int = 15):
        """
        Save a detailed markdown report of the category-only demographic aggregations.

        Args:
            filename (str | Path): Output filename for the report
            top_n (int): Number of top classifications to include per demographic
        """
        filename = self.validate_filename(filename)

        with open(filename, "w") as f:
            # Write header
            f.write("# LDA Taxonomy Category Analysis by Demographics\n\n")
            f.write("## Abstract\n")
            f.write(
                "This report presents the aggregated results of Latent Dirichlet Allocation (LDA) topic modeling applied to blog data, analyzing the most prevalent categories (high-level topics) across different demographic segments. The analysis employs weighted probability scores to ensure accurate representation of category distributions.\n\n"
            )

            f.write("## Methodology\n")
            f.write(
                "The LDA taxonomy classifications were processed by extracting only the category portion from 'category:sub-category' format classifications. These were then aggregated using weighted probability scores where each classification's contribution is proportional to its assigned probability. Demographics were grouped according to the assignment specifications:\n"
            )
            f.write("- Gender: Male and Female\n")
            f.write("- Age: ≤20 years and >20 years\n")
            f.write("- Industry: Students and Non-Students\n")
            f.write("- Overall: Everyone\n\n")

            f.write("## Results\n\n")

            # Write demographic distribution
            f.write("### Demographic Distribution\n\n")
            f.write("| Demographic | Sample Size |\n")
            f.write("|-------------|-------------|\n")

            for demo_name, demo_df in [
                ("Male", self.df.filter(pl.col("gender") == "male")),
                ("Female", self.df.filter(pl.col("gender") == "female")),
                ("Age ≤20", self.df.filter(pl.col("age_group") == "<=20")),
                ("Age >20", self.df.filter(pl.col("age_group") == ">20")),
                ("Students", self.df.filter(pl.col("is_student") == True)),
                ("Non-Students", self.df.filter(pl.col("is_student") == False)),
                ("Everyone", self.df),
            ]:
                f.write(f"| {demo_name} | {demo_df.height} |\n")

            f.write("\n### Top LDA Taxonomy Categories by Demographic\n\n")

            # Write detailed results for each demographic
            for demo_name, classifications in self.category_aggregations.items():
                f.write(f"#### {demo_name}\n\n")
                f.write("| Rank | Category | Weighted Score |\n")
                f.write("|------|----------|----------------|\n")

                top_classifications = self.get_top_classifications(
                    classifications, top_n
                )
                for rank, (classification, score) in enumerate(top_classifications, 1):
                    f.write(f"| {rank} | {classification} | {score:.2f} |\n")

                # Add most popular categories summary
                if len(top_classifications) >= 2:
                    f.write(f"\n**Most Popular Categories for {demo_name}:**\n")
                    f.write(
                        f"1. {top_classifications[0][0]} (Score: {top_classifications[0][1]:.2f})\n"
                    )
                    f.write(
                        f"2. {top_classifications[1][0]} (Score: {top_classifications[1][1]:.2f})\n\n"
                    )

            f.write("## Discussion\n\n")
            f.write(
                "The category-level analysis reveals broad topic preferences across demographic segments. By focusing on high-level categories rather than specific sub-categories, this analysis provides insights into general areas of interest that may inform content strategy and audience targeting.\n\n"
            )

        print(f"Category report saved to {filename}")

    def save_category_subcategory_report(self, filename: str | Path, top_n: int = 15):
        """
        Save a detailed markdown report of the category+sub-category demographic aggregations.

        Args:
            filename (str | Path): Output filename for the report
            top_n (int): Number of top classifications to include per demographic
        """
        filename = self.validate_filename(filename)

        with open(filename, "w") as f:
            # Write header
            f.write("# LDA Taxonomy Category+Sub-Category Analysis by Demographics\n\n")
            f.write("## Abstract\n")
            f.write(
                "This report presents the aggregated results of Latent Dirichlet Allocation (LDA) topic modeling applied to blog data, analyzing the most prevalent specific topics (category:sub-category) across different demographic segments. The analysis employs weighted probability scores to ensure accurate representation of topic distributions.\n\n"
            )

            f.write("## Methodology\n")
            f.write(
                "The LDA taxonomy classifications were aggregated using their full 'category:sub-category' format with weighted probability scores where each classification's contribution is proportional to its assigned probability. This approach provides a more granular view of topic preferences compared to category-only analysis. Demographics were grouped according to the assignment specifications:\n"
            )
            f.write("- Gender: Male and Female\n")
            f.write("- Age: ≤20 years and >20 years\n")
            f.write("- Industry: Students and Non-Students\n")
            f.write("- Overall: Everyone\n\n")

            f.write("## Results\n\n")

            # Write demographic distribution
            f.write("### Demographic Distribution\n\n")
            f.write("| Demographic | Sample Size |\n")
            f.write("|-------------|-------------|\n")

            for demo_name, demo_df in [
                ("Male", self.df.filter(pl.col("gender") == "male")),
                ("Female", self.df.filter(pl.col("gender") == "female")),
                ("Age ≤20", self.df.filter(pl.col("age_group") == "<=20")),
                ("Age >20", self.df.filter(pl.col("age_group") == ">20")),
                ("Students", self.df.filter(pl.col("is_student") == True)),
                ("Non-Students", self.df.filter(pl.col("is_student") == False)),
                ("Everyone", self.df),
            ]:
                f.write(f"| {demo_name} | {demo_df.height} |\n")

            f.write("\n### Top LDA Taxonomy Topics by Demographic\n\n")

            # Write detailed results for each demographic
            for (
                demo_name,
                classifications,
            ) in self.category_subcategory_aggregations.items():
                f.write(f"#### {demo_name}\n\n")
                f.write("| Rank | Topic | Weighted Score |\n")
                f.write("|------|-------|----------------|\n")

                top_classifications = self.get_top_classifications(
                    classifications, top_n
                )
                for rank, (classification, score) in enumerate(top_classifications, 1):
                    f.write(f"| {rank} | {classification} | {score:.2f} |\n")

                # Add most popular topics summary
                if len(top_classifications) >= 2:
                    f.write(f"\n**Most Popular Topics for {demo_name}:**\n")
                    f.write(
                        f"1. {top_classifications[0][0]} (Score: {top_classifications[0][1]:.2f})\n"
                    )
                    f.write(
                        f"2. {top_classifications[1][0]} (Score: {top_classifications[1][1]:.2f})\n\n"
                    )

            f.write("## Discussion\n\n")
            f.write(
                "The detailed category+sub-category analysis provides granular insights into specific topic preferences across demographic segments. This level of detail enables more targeted content development and audience-specific messaging strategies.\n\n"
            )

        print(f"Category+Sub-category report saved to {filename}")
