import pandas as pd
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


class TopicTaxonomyResultsAggregator:
    """
    A class to aggregate topic taxonomy classifications by demographics for academic reporting.

    This class processes topic taxonomy results and aggregates them across different
    demographic segments including gender, age groups, industry (students), and overall population.
    Classifications are weighted by their probability scores to provide more accurate topic
    distributions. The class is designed to be used with the TopicTaxonomyMapper class.
    """

    def __init__(self, parquet_file_path: str):
        """
        Initialize the aggregator with data from a parquet file.

        Args:
            parquet_file_path (str): Path to the parquet file containing LDA results
        """
        self.df = pd.read_parquet(parquet_file_path)
        self.demographic_aggregations = {}
        self._prepare_demographics()
        self._aggregate_by_demographics()

    def _prepare_demographics(self):
        """Prepare demographic groupings based on assignment requirements."""
        # Create age groups: <=20 and >20
        self.df["age_numeric"] = pd.to_numeric(self.df["age"], errors="coerce")
        self.df["age_group"] = self.df["age_numeric"].apply(
            lambda x: "<=20" if pd.notna(x) and x <= 20 else ">20"
        )

        # Create student indicator
        self.df["is_student"] = self.df["industry"] == "Student"

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

    def _aggregate_classifications_for_group(
        self, group_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Aggregate weighted classifications for a demographic group.

        Args:
            group_df (pd.DataFrame): DataFrame subset for the demographic group

        Returns:
            Dict[str, float]: Aggregated weighted classification scores
        """
        aggregated_scores = defaultdict(float)

        for _, row in group_df.iterrows():
            classifications = self._parse_lda_classification_json(
                row["lda_taxonomy_classification"]
            )

            # Add weighted scores for each classification
            for classification, probability in classifications.items():
                aggregated_scores[classification] += (
                    probability / 100.0
                )  # Convert percentage to weight

        return dict(aggregated_scores)

    def _aggregate_by_demographics(self) -> Dict[str, Dict[str, float]]:
        """
        Aggregate classifications across all demographic groups.

        Returns:
            Dict[str, Dict[str, float]]: Nested dictionary with demographics and their classifications
        """
        demographics = {
            "Male": self.df[self.df["gender"] == "male"],
            "Female": self.df[self.df["gender"] == "female"],
            "Age <=20": self.df[self.df["age_group"] == "<=20"],
            "Age >20": self.df[self.df["age_group"] == ">20"],
            "Students": self.df[self.df["is_student"] == True],  # noqa: E712
            "Non-Students": self.df[self.df["is_student"] == False],  # noqa: E712
            "Everyone": self.df,
        }

        results = {}
        for demo_name, demo_df in demographics.items():
            if not demo_df.empty:
                results[demo_name] = self._aggregate_classifications_for_group(demo_df)

        self.demographic_aggregations = results
        return results

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

    def generate_academic_report_lda(self, top_n: int = 10) -> str:
        """
        Generate an academic report of the LDA taxonomy classification results.

        Args:
            top_n (int): Number of top classifications to include per demographic

        Returns:
            str: Formatted academic report
        """
        report = []
        report.append("# LDA Taxonomy Classification Analysis by Demographics")
        report.append("")
        report.append("## Abstract")
        report.append(
            "This report presents the aggregated results of Latent Dirichlet Allocation (LDA) "
            "topic modeling applied to blog data, analyzing the most prevalent topics across "
            "different demographic segments. The analysis employs weighted probability scores "
            "to ensure accurate representation of topic distributions."
        )
        report.append("")

        report.append("## Methodology")
        report.append(
            "The LDA taxonomy classifications were aggregated using weighted probability scores "
            "where each classification's contribution is proportional to its assigned probability. "
            "This approach provides a more nuanced view of topic prevalence compared to simple "
            "frequency counting. Demographics were grouped according to the assignment specifications:"
        )
        report.append("- Gender: Male and Female")
        report.append("- Age: ≤20 years and >20 years")
        report.append("- Industry: Students and Non-Students")
        report.append("- Overall: Everyone")
        report.append("")

        report.append("## Results")
        report.append("")

        # Generate demographic counts table
        demo_counts = {
            "Male": len(self.df[self.df["gender"] == "male"]),
            "Female": len(self.df[self.df["gender"] == "female"]),
            "Age ≤20": len(self.df[self.df["age_group"] == "<=20"]),
            "Age >20": len(self.df[self.df["age_group"] == ">20"]),
            "Students": len(self.df[self.df["is_student"] == True]),  # noqa: E712
            "Non-Students": len(self.df[self.df["is_student"] == False]),  # noqa: E712
            "Everyone": len(self.df),
        }

        report.append("### Demographic Distribution")
        report.append("")
        report.append("| Demographic | Sample Size |")
        report.append("|-------------|-------------|")
        for demo, count in demo_counts.items():
            report.append(f"| {demo} | {count} |")
        report.append("")

        # Generate top classifications for each demographic
        report.append("### Top LDA Taxonomy Classifications by Demographic")
        report.append("")

        for demo_name, classifications in self.demographic_aggregations.items():
            report.append(f"#### {demo_name}")
            report.append("")

            top_classifications = self.get_top_classifications(classifications, top_n)

            if top_classifications:
                report.append("| Rank | Classification | Weighted Score |")
                report.append("|------|----------------|----------------|")

                for rank, (classification, score) in enumerate(top_classifications, 1):
                    report.append(f"| {rank} | {classification} | {score:.2f} |")
                report.append("")

                # Highlight top 2 most popular topics
                if len(top_classifications) >= 2:
                    report.append(f"**Most Popular Topics for {demo_name}:**")
                    report.append(
                        f"1. {top_classifications[0][0]} (Score: {top_classifications[0][1]:.2f})"
                    )
                    report.append(
                        f"2. {top_classifications[1][0]} (Score: {top_classifications[1][1]:.2f})"
                    )
                    report.append("")
            else:
                report.append("No classifications found for this demographic.")
                report.append("")

        report.append("## Discussion")
        report.append("")
        report.append(
            "The weighted aggregation approach reveals distinct topic preferences across "
            "demographic segments. The scoring methodology ensures that topics with higher "
            "LDA probability assignments contribute more significantly to the final rankings, "
            "providing a more accurate representation of topic prevalence than simple counting."
        )
        report.append("")

        report.append("### Key Observations:")
        report.append("")

        # Generate comparative insights
        top_2_by_demo = {}
        for demo_name, classifications in self.demographic_aggregations.items():
            top_2 = self.get_top_classifications(classifications, 2)
            if len(top_2) >= 2:
                top_2_by_demo[demo_name] = [t[0] for t in top_2]

        report.append("**Top 2 Topics by Demographic:**")
        for demo, topics in top_2_by_demo.items():
            if len(topics) >= 2:
                report.append(f"- **{demo}**: {topics[0]}, {topics[1]}")
        report.append("")

        report.append("## Conclusion")
        report.append("")
        report.append(
            "This analysis successfully identified the two most popular topics across the "
            "specified demographic segments using weighted LDA taxonomy classifications. "
            "The methodology provides a robust foundation for understanding topic prevalence "
            "in blog data while accounting for the probabilistic nature of LDA topic assignments."
        )

        return "\n".join(report)

    def validate_filename(self, filename: str | Path):
        """
        Validate the filename.
        """
        if isinstance(filename, str):
            filename = Path(filename)

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        return filename

    def save_report(self, filename: str | Path, top_n: int = 10):
        """
        Save the academic report to a markdown file.

        Args:
            filename (str): Output filename
            top_n (int): Number of top classifications to include
        """
        filename = self.validate_filename(filename)

        report = self.generate_academic_report_lda(top_n)

        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to {filename}")

    def save_demographics_to_parquet(self, filename: str | Path):
        """
        Save the demographic aggregations to a parquet file.

        Args:
            filename (str): Output filename for the parquet file
        """
        filename = self.validate_filename(filename)

        # Convert aggregations to a structured format suitable for parquet
        records = []
        for demographic, classifications in self.demographic_aggregations.items():
            for position, (classification, weighted_score) in enumerate(
                classifications.items()
            ):
                records.append({
                    "demographic": demographic,
                    "position": position,
                    "classification": classification,
                    "weighted_score": weighted_score,
                })

        # Create DataFrame and save to parquet
        df_results = pd.DataFrame(records)
        df_results.to_parquet(filename, index=False)
        print(f"Demographic aggregations saved to {filename}")

    def get_demographics_dataframe(self) -> pd.DataFrame:
        """
        Get the demographic aggregations as a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame with columns [demographic, classification, weighted_score]
        """
        records = []
        for demographic, classifications in self.demographic_aggregations.items():
            for classification, weighted_score in classifications.items():
                records.append({
                    "demographic": demographic,
                    "classification": classification,
                    "weighted_score": weighted_score,
                })

        return pd.DataFrame(records)

    def get_demographic_summary(self) -> Dict[str, Tuple[str, str]]:
        """
        Get a summary of the top 2 topics for each demographic.

        Returns:
            Dict[str, Tuple[str, str]]: Dictionary mapping demographics to their top 2 topics
        """
        summary = {}
        for demo_name, classifications in self.demographic_aggregations.items():
            top_2 = self.get_top_classifications(classifications, 2)
            if len(top_2) >= 2:
                summary[demo_name] = (top_2[0][0], top_2[1][0])
            elif len(top_2) == 1:
                summary[demo_name] = (top_2[0][0], "No second topic")
            else:
                summary[demo_name] = ("No topics found", "No topics found")

        return summary
