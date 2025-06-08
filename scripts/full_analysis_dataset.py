import json
import logging
from logging import getLogger
from pathlib import Path
import time
import gc
import psutil
from datetime import datetime
import argparse
import sys
import shutil

import polars as pl

from topic_extractor.data_extraction import BlogDataProcessor
from topic_extractor.data_transformation import PostsTableTransformation
from topic_extractor.lda_tranformer_extractor import TransformerEnhancedLDA
from topic_extractor.results_aggregation import TopicTaxonomyResultsAggregator
from topic_extractor.topic_simplifying import (
    TopicTaxonomyMapper,
    map_lda_results_to_taxonomy,
)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
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


def process_lda_with_progress(
    df: pl.LazyFrame,
    final_output_path: Path,
    batch_size: int = 50,
    output_base_path: str = ".data/tables/lda_extracted_batches",
    save_frequency: int = 5,
) -> pl.LazyFrame:
    """
    Process LDA with enhanced batch processing, progress tracking, and partitioned output.

    Args:
        df: Input LazyFrame with content to process
        final_output_path: Path to save the final combined result
        batch_size: Number of records per batch
        output_base_path: Base path for saving batch results
        save_frequency: Save intermediate results every N batches

    Returns:
        LazyFrame with LDA results
    """
    start_time = time.time()

    # Collect the data to work with concrete operations
    logger.info("Collecting data for batch processing...")
    data = df.collect()
    total_records = len(data)

    logger.info(f"Starting enhanced LDA processing for {total_records} records")
    logger.info(
        f"Batch size: {batch_size}, Save frequency: every {save_frequency} batches"
    )

    # Create output directory
    output_path = Path(output_base_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize LDA model once
    logger.info("Initializing LDA model...")
    lda_obj = TransformerEnhancedLDA(min_topic_size=5)

    # Process in batches
    all_results = []
    batch_results = []

    num_batches = (total_records + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_start_time = time.time()

        # Calculate batch boundaries
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_records)

        # Extract batch
        batch_data = data.slice(start_idx, end_idx - start_idx)

        logger.info(
            f"Processing batch {batch_idx + 1}/{num_batches} (records {start_idx + 1}-{end_idx})"
        )

        # Process LDA for this batch
        try:
            batch_with_lda = batch_data.with_columns(
                pl.col("content")
                .map_batches(
                    lambda x: process_lda_batch(x, lda_obj),
                    return_dtype=pl.Utf8,
                )
                .alias("lda_topics"),
            )

            batch_results.append(batch_with_lda)

            # Memory management
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time

            logger.info(
                f"Batch {batch_idx + 1} completed in {batch_duration:.2f}s "
                f"({(end_idx - start_idx) / batch_duration:.1f} records/sec)"
            )

            # Save intermediate results periodically
            if (batch_idx + 1) % save_frequency == 0 or batch_idx == num_batches - 1:
                # Combine processed batches
                if len(batch_results) > 1:
                    combined_batch = pl.concat(batch_results, rechunk=True)
                else:
                    combined_batch = batch_results[0]

                # Save partition
                partition_path = (
                    output_path
                    / f"batch_{batch_idx + 1 - len(batch_results) + 1}_to_{batch_idx + 1}.parquet"
                )
                logger.info(f"Saving partition to {partition_path}")
                combined_batch.write_parquet(partition_path, compression="snappy")

                # Add to all results and clear batch results for memory
                all_results.append(combined_batch)
                batch_results = []

                # Force garbage collection
                gc.collect()

                # Progress summary with memory monitoring
                elapsed_time = time.time() - start_time
                records_processed = end_idx
                records_remaining = total_records - records_processed
                avg_speed = records_processed / elapsed_time
                eta_seconds = records_remaining / avg_speed if avg_speed > 0 else 0

                # Memory usage
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_percent = process.memory_percent()

                logger.info(
                    f"Progress: {records_processed}/{total_records} "
                    f"({100 * records_processed / total_records:.1f}%) | "
                    f"Speed: {avg_speed:.1f} records/sec | "
                    f"ETA: {eta_seconds / 60:.1f} minutes | "
                    f"Memory: {memory_mb:.1f}MB ({memory_percent:.1f}%)"
                )

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx + 1}: {str(e)}")
            # Create error batch with empty LDA results
            error_batch = batch_data.with_columns(
                pl.lit(
                    json.dumps({
                        "lda_results": {"topics": []},
                        "words": set(),
                        "topic_labels": [],
                        "error": str(e),
                    })
                ).alias("lda_topics")
            )
            batch_results.append(error_batch)
            continue

    # Combine all results
    logger.info("Combining all processed batches...")
    if len(all_results) > 1:
        final_result = pl.concat(all_results, rechunk=True)
    else:
        final_result = all_results[0]

    total_time = time.time() - start_time
    logger.info(
        f"LDA processing completed! Total time: {total_time:.2f}s "
        f"({total_records / total_time:.1f} records/sec)"
    )

    # Save final combined result
    logger.info(f"Saving final combined result to {final_output_path}")
    final_result.write_parquet(final_output_path, compression="snappy", statistics=True)

    return final_result.lazy()


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

    # Number of blogs to process (0 for all) - using small number to show progress tracking
    keep_rows = 0

    # Path for processed raw data
    files_df_path = Path(".data/tables/files_df.parquet")
    posts_df_path = Path(".data/tables/posts_df.parquet")

    # Path for the combined data
    blogs_full_df_path = Path(".data/tables/blogs_full_df.parquet")

    # Enhanced LDA processing with progress tracking and partitioned output
    enhanced_lda_path = Path(f".data/tables/lda_extracted_df_final_{keep_rows}.parquet")

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
        blogs_english_transformed_df = pl.scan_parquet(blogs_full_df_path)
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
        blogs_english_transformed_df.sink_parquet(
            path=str(blogs_full_df_path.resolve()), statistics=True, mkdir=True
        )

    # If we want to keep a limited number of blogs, we can do so here
    if keep_rows > 0:
        blogs_english_transformed_df = blogs_english_transformed_df.limit(keep_rows)

    if enhanced_lda_path.exists():
        logger.info("Loading enhanced LDA results from cache")
        blogs_lda_extracted_df = pl.scan_parquet(enhanced_lda_path)

    if not enhanced_lda_path.exists():
        logger.info("Starting enhanced LDA processing with progress tracking")

        # Use enhanced LDA processing with progress tracking
        blogs_lda_extracted_df = process_lda_with_progress(
            blogs_english_transformed_df,
            final_output_path=enhanced_lda_path,
            batch_size=10,  # Small batches to show progress clearly
            output_base_path=f".data/tables/lda_extracted_batches_{keep_rows if keep_rows > 0 else 'all'}",
            save_frequency=2,  # Save every 2 batches for frequent updates
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
