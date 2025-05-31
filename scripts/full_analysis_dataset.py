from pathlib import Path
import json
import polars as pl

from topic_extractor.data_transformation import PostsTableTransformation
from topic_extractor.lda_tranformer_extractor import TransformerEnhancedLDA
from topic_extractor.data_extraction import BlogDataProcessor
from topic_extractor.topic_simplifying import TopicTaxonomyMapper
from logging import getLogger


logger = getLogger(__name__)


def main():
    """
    This script is a placeholder for providing the full process to be replicated in a notebook for the assignment.
    """

    # Path for processed data
    files_df_path = Path(".data/tables/files_df.parquet")
    posts_df_path = Path(".data/tables/posts_df.parquet")

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

    # Join the files_df with the posts_df
    full_df = posts_df.join(files_df, left_on="file_id", right_on="id", how="left")

    # Keep only a random sample
    keep_rows = 200
    logger.info(f"Keeping only a random sample of {keep_rows} rows")
    full_df = full_df.limit(keep_rows)

    # Next we need to transform the data to be used for the LDA model
    transformer = PostsTableTransformation(full_df)
    transformed_df = (
        transformer.detect_language()
        .compute_word_frequencies(n_most_common=5, n_least_common=3)
        .get_dataframe()
    )

    # Keep only the english posts
    english_transformed_df = transformed_df.filter(pl.col("content_language") == "en")

    # Now we start to extract the topics in different ways
    # Apply the LDA model to the transformed data's content column
    lda_obj = TransformerEnhancedLDA(min_topic_size=5)
    taxonomy_mapper = TopicTaxonomyMapper()

    lda_extracted_df = english_transformed_df.with_columns(
        pl.col("content")
        .map_elements(
            lambda x: json.dumps(lda_obj.extract_topics(x), default=str),
            return_dtype=pl.Utf8,
        )
        .alias("lda_topics"),
    )

    lda_parsed_df = lda_extracted_df.with_columns(
        pl.col("lda_topics")
        .map_elements(
            lambda x: json.loads(x).get("words"),
            return_dtype=pl.Utf8,
        )
        .alias("lda_topic_words"),
    )

    lda_taxonomy_df = lda_parsed_df.with_columns(
        pl.col("lda_topic_words")
        .map_elements(
            lambda x: json.dumps(
                taxonomy_mapper.map_words_to_taxonomy(x, top_n=10), default=str
            ),
            return_dtype=pl.Utf8,
        )
        .alias("lda_taxonomy_classification"),
    )

    lda_taxonomy_df.sink_parquet(
        path=".data/tables/lda_taxonomy_df.parquet", statistics=True, mkdir=True
    )


if __name__ == "__main__":
    main()
