import json
import logging
from logging import getLogger
from pathlib import Path

import polars as pl

from topic_extractor.data_extraction import BlogDataProcessor
from topic_extractor.data_transformation import PostsTableTransformation
from topic_extractor.lda_tranformer_extractor import TransformerEnhancedLDA
from topic_extractor.results_aggregation import TopicTaxonomyResultsAggregator
from topic_extractor.topic_simplifying import (
    TopicTaxonomyMapper,
    map_lda_results_to_taxonomy,
)

logging.basicConfig(level=logging.DEBUG)
logger = getLogger(__name__)


def process_lda_batch(series: pl.Series, lda_obj: TransformerEnhancedLDA) -> pl.Series:
    """
    Process a batch of texts through LDA.
    Returns a Series of JSON strings with the same length as input.
    """
    # Convert to list for processing
    texts = series.to_list()

    # Process each text
    results = []
    for text in texts:
        lda_result = lda_obj.extract_topics(text)
        results.append(json.dumps(lda_result, default=str))

    # Return as Series with same length
    return pl.Series(results)


def process_taxonomy_batch(
    series: pl.Series, taxonomy_mapper: TopicTaxonomyMapper
) -> pl.Series:
    """
    Process taxonomy mapping for a batch.
    Returns a Series of JSON strings with the same length as input.
    """
    lda_results = series.to_list()

    results = []
    for lda_json in lda_results:
        lda_data = json.loads(lda_json)
        taxonomy_result = map_lda_results_to_taxonomy(taxonomy_mapper, lda_data)
        results.append(json.dumps(taxonomy_result))

    return pl.Series(results)


def create_blogs_df(posts_df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Create a Blog Posts LazyFrame by grouping posts by blog (file_id)
    """
    # Create LDA-optimized dataframe by grouping posts by blog (file_id)
    logger.info("Creating LDA-optimized dataframe by grouping posts per blog")
    blog_posts_df = (
        posts_df
        # Sort by file_id and date if available to maintain chronological order
        .sort(
            ["file_id", "date"]
            if "date" in posts_df.collect_schema().names()
            else ["file_id"]
        )
        # Group by file_id to combine all posts from the same blog
        .group_by("file_id")
        .agg([
            # Concatenate all content with double newline separator for clean separation
            pl.col("content").str.concat(delimiter="\n\n").alias("content"),
            # Keep useful metadata
            pl.col("content").len().alias("post_count"),
            pl.col("content").str.len_chars().sum().alias("total_content_length"),
            # If date column exists, get date range
            *(
                [
                    pl.col("date").min().alias("earliest_post_date"),
                    pl.col("date").max().alias("latest_post_date"),
                ]
                if "date" in posts_df.collect_schema().names()
                else []
            ),
        ])
    )

    return blog_posts_df


def main():
    """
    This script is a placeholder for providing the full process to be replicated in a notebook for the assignment.
    """

    # Number of blogs to process (0 for all)
    keep_rows = 0

    # Path for processed raw data
    files_df_path = Path(".data/tables/files_df.parquet")
    posts_df_path = Path(".data/tables/posts_df.parquet")

    # Path for the combined data
    blogs_full_df_path = Path(".data/tables/blogs_full_df.parquet")

    # Path for the LDA-extracted data
    lda_extracted_df_path = Path(f".data/tables/lda_extracted_df_{keep_rows}.parquet")

    # Path for the LDA-taxonomy data
    lda_taxonomy_df_path = Path(f".data/tables/lda_taxonomy_df_{keep_rows}.parquet")

    # First lets check if the data has already been processed, for faster development
    if posts_df_path.exists() and files_df_path.exists():
        logger.info("Loading processed data from cache")
        files_df = pl.scan_parquet(files_df_path)
        posts_df = pl.scan_parquet(posts_df_path)
    else:
        logger.info("No processed data found, extracting data from xml files")
        # First we need to extract the data from the xml files
        data_processor = BlogDataProcessor(
            data_directory=str(Path(".data/blogs").resolve())
        )
        files_df, posts_df = data_processor.create_dataframes()

        # Save the data to the path
        data_processor.save_lazyframe(files_df, str(files_df_path.resolve()))
        data_processor.save_lazyframe(posts_df, str(posts_df_path.resolve()))

    if blogs_full_df_path.exists():
        logger.info("Loading blogs_full_df from cache")
        blogs_transformer = pl.scan_parquet(blogs_full_df_path)
    else:
        logger.info("No blogs_full_df found, creating it")

        # Prepare an LDA-optimized dataframe
        blog_posts_df = create_blogs_df(posts_df)

        # Join the files_df with the lda_post_df (grouped posts)
        blogs_full_df = blog_posts_df.join(
            files_df, left_on="file_id", right_on="id", how="left"
        )
        # Next we need to prepare the data for the topic extraction models
        blogs_transformer = PostsTableTransformation(blogs_full_df)
        blogs_transformer = (
            blogs_transformer.detect_language()
            .clean_up_content_column()
            .clean_up_industry_column()
            .get_dataframe()
        )

        # Keep only the english posts for focused analysis
        blogs_english_transformed_df = blogs_transformer.filter(
            pl.col("content_language") == "en"
        )

        # Save the data to the path
        blogs_transformer.sink_parquet(
            path=str(blogs_full_df_path.resolve()), statistics=True, mkdir=True
        )

    # If we want to keep a limited number of blogs, we can do so here
    if keep_rows > 0:
        blogs_full_df = blogs_full_df.limit(keep_rows)

    if lda_extracted_df_path.exists():
        logger.info("Loading lda_extracted_df from cache")
        blogs_lda_extracted_df = pl.scan_parquet(lda_extracted_df_path)
    else:
        logger.info("No lda_extracted_df found, creating it")

        # Now we start to extract the topics in different ways
        # Apply the LDA model to the transformed data's content column
        lda_obj = TransformerEnhancedLDA(min_topic_size=5)

        blogs_lda_extracted_df = blogs_english_transformed_df.with_columns(
            pl.col("content")
            .map_batches(
                lambda x: process_lda_batch(x, lda_obj),
                return_dtype=pl.Utf8,
            )
            .alias("lda_topics"),
        )

        # Save the lda_extracted_df to a parquet file
        blogs_lda_extracted_df.sink_parquet(
            path=str(lda_extracted_df_path.resolve()), statistics=True, mkdir=True
        )

    if lda_taxonomy_df_path.exists():
        logger.info("Loading lda_taxonomy_df from cache")
        blogs_lda_taxonomy_df = pl.scan_parquet(lda_taxonomy_df_path)
    else:
        taxonomy_mapper = TopicTaxonomyMapper()
        logger.info("No lda_taxonomy_df found, creating it")

        # Now we can map the extracted topics to the taxonomy
        blogs_lda_taxonomy_df = blogs_lda_extracted_df.with_columns(
            pl.col("lda_topics")
            .map_batches(
                lambda x: process_taxonomy_batch(x, taxonomy_mapper),
                return_dtype=pl.Utf8,
            )
            .alias("lda_taxonomy_classification"),
        )

        blogs_lda_taxonomy_df.sink_parquet(
            path=str(lda_taxonomy_df_path.resolve()), statistics=True, mkdir=True
        )

    # Once saved, we can run the aggregation and analysis
    lda_aggregator = TopicTaxonomyResultsAggregator(
        parquet_file_path=str(lda_taxonomy_df_path.resolve())
    )
    lda_aggregator.save_category_demographics_to_parquet(
        filename=".data/tables/lda/category_demographics_aggregated.parquet"
    )
    lda_aggregator.save_category_subcategory_demographics_to_parquet(
        filename=".data/tables/lda/category_subcategory_demographics_aggregated.parquet"
    )
    lda_aggregator.save_biased_category_demographics_to_parquet(
        filename=".data/tables/lda/biased_category_demographics_aggregated.parquet"
    )
    lda_aggregator.save_biased_category_subcategory_demographics_to_parquet(
        filename=".data/tables/lda/biased_category_subcategory_demographics_aggregated.parquet"
    )


if __name__ == "__main__":
    main()
