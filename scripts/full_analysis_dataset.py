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
    transformed_df.with_columns(
        pl.col("content")
        .map_elements(
            lambda x: TransformerEnhancedLDA(x, min_topic_size=5).extract_topics(),
            return_dtype=pl.List(
                pl.Struct([
                    pl.Field("topic_id", pl.Int64),
                    pl.Field("topic_name", pl.Utf8),
                    pl.Field("topic_score", pl.Float64),
                ])
            ),
        )
        .alias("lda_topics")
    )

    # Join the transformed dataframe with files_df to get the metadata
    joined_df = posts_df.join(files_df, left_on="file_id", right_on="id", how="left")


if __name__ == "__main__":
    main()
