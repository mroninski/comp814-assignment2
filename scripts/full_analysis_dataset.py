import json
import polars as pl

from topic_extractor.data_transformation import PostsTableTransformation
from topic_extractor.lda_tranformer_extractor import TransformerEnhancedLDA
from topic_extractor.data_extraction import BlogDataProcessor


def main():
    """
    This script is a placeholder for providing the full process to be replicated in a notebook for the assignment.
    """

    # First we need to extract the data from the xml files
    data_processor = BlogDataProcessor(data_directory="../.data/blogs")
    files_df, posts_df = data_processor.create_dataframes()

    # Keep only a random sample of 100 rows
    posts_df = posts_df.limit(100)

    # Next we need to transform the data to be used for the LDA model
    transformer = PostsTableTransformation(posts_df)
    transformed_df = (
        transformer.detect_language()
        .compute_word_frequencies(n_most_common=5, n_least_common=3)
        .get_dataframe()
    )

    # Now we start to extract the topics in different ways
    # Apply the LDA model to the transformed data's content column
    lda_obj = TransformerEnhancedLDA(min_topic_size=5)
    transformed_df.with_columns(
        pl.col("content")
        .map_elements(
            lambda x: json.dumps(lda_obj.extract_topics(x)),
            return_dtype=pl.Utf8,
        )
        .alias("lda_topics"),
        pl.col("lda_topics")
        .map_elements(
            lambda x: json.loads(x)["words"],
            return_dtype=pl.List(pl.Utf8),
        )
        .alias("lda_topic_words"),
    )


if __name__ == "__main__":
    main()
