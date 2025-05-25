import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, Optional

import polars as pl
from bs4 import BeautifulSoup
from datetime import datetime
import dateparser


class BlogDataProcessor:
    def __init__(self, data_directory: str):
        """
        Initialize the blog data processor.

        Args:
            data_directory: Path to directory containing XML blog files
        """
        self.data_directory = data_directory

    def _parse_multilingual_date(self, date_string: str) -> Optional[float]:
        """
        Parse dates with month names in multiple languages using dateutil.
        This is a fuzzy method, so the precision might be slightly low but this is a low priority task.

        Args:
            date_string: Date string in format "day,month_name,year"

        Returns:
            datetime object if parsing succeeds, None otherwise
        """
        try:
            # Replace commas with spaces for better parsing
            normalized_date = date_string.replace(",", " ").strip()

            # Use dateutil's fuzzy parser which can handle many different formats
            # and languages automatically
            parsed_date = dateparser.parse(
                normalized_date, settings={"DATE_ORDER": "DMY"}
            )

            if parsed_date is None:
                return None

            return parsed_date.timestamp()

        except (ValueError, TypeError) as e:
            print(f"Error parsing date '{date_string}': {e}")
            return None

    def files_dataframe(self) -> pl.LazyFrame:
        """
        Create a dataframe with metadata from filenames.

        Returns:
            LazyFrame with file metadata and IDs
        """
        # Get all XML files
        file_paths = Path(self.data_directory).glob("*.xml")

        # Create initial dataframe with file paths
        files_df = pl.LazyFrame({"file_path": file_paths})

        def extract_filename(file_path: Path) -> str:
            return str(file_path.stem)

        # Extract filename without path
        files_df = files_df.with_columns(
            pl.col("file_path")
            .map_elements(extract_filename, return_dtype=pl.Utf8)
            .alias("filename")
        ).collect()

        # Extract metadata from filenames
        files_df = files_df.with_columns([
            pl.col("filename").str.split(".").list.get(0).alias("scrapping_timestamp"),
            pl.col("filename").str.split(".").list.get(1).alias("gender"),
            pl.col("filename").str.split(".").list.get(2).alias("age"),
            pl.col("filename").str.split(".").list.get(3).alias("industry"),
            pl.col("filename")
            .str.split(".")
            .list.get(4)
            .str.replace(r"\.xml$", "")
            .alias("star_sign"),
            pl.col("file_path")
            .map_elements(lambda x: str(x.resolve()), return_dtype=pl.Utf8)
            .alias("file_path"),
        ])

        # Generate ID by hashing concatenated metadata
        files_df = files_df.with_columns(
            pl.concat_list(
                pl.col("scrapping_timestamp"),
                pl.col("gender"),
                pl.col("age"),
                pl.col("industry"),
                pl.col("star_sign"),
            )
            .map_elements(lambda s: hash(tuple(s)), return_dtype=pl.Int64)
            .alias("id")
        )

        # Select final columns
        files_df = files_df.select([
            "id",
            "scrapping_timestamp",
            "gender",
            "age",
            "industry",
            "star_sign",
            "file_path",
        ])

        return pl.LazyFrame(files_df)

    def _safe_parse_xml(self, file_path: str) -> list[dict[str, str]]:
        """
        Safely parse XML file with BeautifulSoup to handle malformed XML.

        Args:
            file_path: Path to XML file

        Returns:
            List of dictionaries with date and content for each post
        """
        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                file_content = f.read()

            # Fix common XML issues - replace unescaped ampersands
            file_content = re.sub(
                r"&(?!amp;|lt;|gt;|apos;|quot;)", "&amp;", file_content
            )

            # Parse with BeautifulSoup which is more forgiving
            soup = BeautifulSoup(file_content, "xml")

            # Extract date and post elements
            posts_data = []
            date_elements = soup.find_all("date")
            post_elements = soup.find_all("post")

            # Process each date-post pair
            for date_elem, post_elem in zip(date_elements, post_elements):
                date = date_elem.text.strip() if date_elem.text else ""
                content = post_elem.text.strip() if post_elem.text else ""

                posts_data.append({"date": date, "content": content})

            return posts_data

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return []

    def posts_dataframe(self, files_df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Create a dataframe with blog posts from XML files.

        Args:
            files_df: LazyFrame with file metadata

        Returns:
            LazyFrame with post data
        """
        # Materialize files dataframe to get file paths
        files_data = files_df.collect()

        # Process XML files to extract posts
        posts_data = []

        for row in files_data.iter_rows(named=True):
            file_id: str = row["id"]
            file_path: str = row["file_path"]

            # Parse XML file safely
            file_posts = self._safe_parse_xml(file_path)

            # Add file_id to each post
            for post in file_posts:
                post["file_id"] = file_id
                posts_data.append(post)

        # Create posts dataframe
        return pl.LazyFrame(posts_data)

    def _transform_posts_dataframe(self, posts_df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Transform the posts dataframe to include the date as a datetime object.
        """
        return posts_df.with_columns(
            pl.col("date")
            .map_elements(self._parse_multilingual_date, return_dtype=pl.Float64)
            .alias("date")
        )

    def save_lazyframe(self, lazyframe: pl.LazyFrame, path: str) -> None:
        """
        Save the lazyframe to a parquet file.
        """
        lazyframe.sink_parquet(path=path, statistics=True, mkdir=True)

    def create_dataframes(self) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """
        Create files and posts dataframes.

        Returns:
            Tuple of (files_df, posts_df)
        """
        # Create files dataframe
        files_df = self.files_dataframe()

        # Create posts dataframe
        raw_posts_df = self.posts_dataframe(files_df)

        # Transform posts dataframe
        posts_df = self._transform_posts_dataframe(raw_posts_df)

        return files_df, posts_df


if __name__ == "__main__":
    data_processor = BlogDataProcessor(
        data_directory="/Users/t834527/Repos/comp814-assignment2/.data/blogs"
    )
    files_df, posts_df = data_processor.create_dataframes()

    # Save the dataframes to parquet files
    data_processor.save_lazyframe(files_df, ".data/tables/files_df")
    data_processor.save_lazyframe(posts_df, ".data/tables/posts_df")

    # Join the two dataframes on the file_id column
    joined_df = posts_df.join(files_df, left_on="file_id", right_on="id", how="left")

    # Save the dataframe to a parquet file
    joined_df.sink_parquet(path=".data/tables/joined_df", statistics=True, mkdir=True)
