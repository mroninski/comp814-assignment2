## 8. Results and Analysis

Our analysis presents findings first through a simplified topic distribution, followed by demographic-specific results utilizing the Demographic Specificity Score (DSS) to identify category-specific patterns that are more prominent on each demographic individually.

### 8.1 Overall Topic Distribution

#### Transformer-LDA Results

The average weighted analysis identified **entertainment** as the predominant topic across all demographic segments, with scores ranging from 2.073 to 2.154. Any score greater than 1.0 indicates high relevance. The **video_gaming** topic maintained consistent secondary positioning across demographics, with scores between 1.458 and 1.534.

In the subcategory level, **technology_and_computing:it and internet support** is the highest weight topic across all demographics, achieving average weighted scores of approximately 0.67. The **video_gaming:music and party video games** subcategory was ranked second with scores around 0.64.

The subcategory-level analysis highlights the limitations of the average weighted analysis. Common high-relevance terms influenced average scores toward specific category clusters within the vector space. This phenomenon likely contributed to the ranking of **video_gaming:music and party video games**, which does not accurately reflect its actual popularity within the corpus.

#### TF-IDF Results

*Note: TF-IDF analysis results are not included in the provided dataset and require separate computational analysis for comparative presentation.*

### 8.2 Demographic-Specific Results

#### 8.2.1 Gender-Based Analysis

##### Transformer-LDA Results

Female bloggers had the highest affinity for **personal_celebrations_and_life_events** (DSS score: 1.314), indicating significantly higher likelihood of documenting personal milestones, celebrations, and life transitions. The **style_and_fashion** topic (DSS score: 1.125).

Within subcategories, female bloggers showed pronounced preference for **healthy_living:weight loss** (DSS score: 2.000). The **hobbies_and_interests:jewelry making** (DSS score: 2.000) and **personal_celebrations_and_life_events:wedding** (DSS score: 2.000) subcategories exhibited similar score proximity and themese, suggesting that they were discussed in the same posts.

Male bloggers had  **religion_and_spirituality** achieving the highest DSS score (2.000). Manual analysis suggested this is related to discourse surrounding the Iraq War, which occurred during the dataset's temporal scope. **Pop_culture** ranked second for male bloggers (DSS score: 1.665).

Male subcategory preferences included **video_gaming:mobile games** (DSS score: 1.999), **video_gaming:educational video games** (DSS score: 1.999), and **video_gaming:pc games** (DSS score: 1.999). These results show a clear thematic clustering around gaming-related content. It must be noted, that mobile gaming category is likely incorrectly categorised, when considering the limited mobile gaming landscape of the early 2000s.

##### TF-IDF Results

*TF-IDF gender-based analysis results are not available in the current dataset.*

#### 8.2.2 Age-Based Analysis

##### Transformer-LDA Results

Bloggers aged 20 and under demonstrated highest affinity for **personal_celebrations_and_life_events** (DSS score: 1.407), suggesting blogs were used to document experiences and milestones. **Education** (DSS score: 1.243) ranked second, reflecting the role of school in their personal discussions.

The subcategory analysis for under-20 users revealed **style_and_fashion:oral care** (DSS score: 2.343) and **personal_finance:personal taxes** (DSS score: 2.343) as most distinctive. These results highlight a methodological limitation where the DSS approach overvalued infrequently occurring but demographically concentrated topics. For example, the **oral care** subcategory appeared only four times total, but its exclusive occurrence within the under-20 demographic resulted in the disproportionate high scoring.

Bloggers over 20 years demonstrated highest engagement with **attractions** (DSS score: 1.415), encompassing travel experiences, entertainment venues, and cultural attractions. **Politics** (DSS score: 1.374) ranked second, indicating greater political discourse engagement among older bloggers.

Over-20 subcategory most demographically unique topics were **business_and_finance:information services industry** (DSS score: 1.744) and **business_and_finance:mechanical and industrial engineering industry** (DSS score: 1.744). These patterns align with expected employment demographics, considering the technology sector's prominence among blog users of the era.

##### TF-IDF Results

*TF-IDF age-based analysis results are not available in the current dataset.*

#### 8.2.3 Student Population

##### Transformer-LDA Results

Students exhibited **religion_and_spirituality** as their most distinctive category (DSS score: 2.000), with **personal_celebrations_and_life_events** ranking second (DSS score: 1.256). Given the substantial overlap between students and under-20 demographics (under-20 students: 4,202; over-20 students: 907), these results demonstrate cross-categorical pattern consistency.

The highest-ranking student subcategory was **religion_and_spirituality:christianity** (DSS score: 3.773). This classification likely reflects linguistic behavior patterns, such as colloquial usage of religious terms as expressions, rather than substantive religious discourse. **Personal_finance:personal taxes** ranked second (DSS score: 3.773).

Non-students demonstrated **fine_art** (DSS score: 1.240) as their top-ranked topic, followed by **pets** (DSS score: 1.143). Non-student subcategory preferences included **entertainment:urban ac music** (DSS score: 1.361) and **hobbies_and_interests:content production** (DSS score: 1.361).

##### TF-IDF Results

*TF-IDF student population analysis results are not available in the current dataset.*

### 8.3 Method Comparison

#### TF-IDF vs. Transformer-LDA Performance
*TBC* 

#### Topic Quality and Coherence Scores

The dual scoring framework provided complementary analytical perspectives that improved the result interpretation. While average weighted scoring identified the most prevalent topics generally, the high magnitude of these scores obscured demographic-specific nuances. The Demographic Specific Results approach revealed preference patterns and behaviors that, while consistent with existing research, provided additional analytical depth. This methodological approach addressed the challenge of identifying minority preferences within large, heterogeneous datasets.

#### Computational Efficiency Analysis

The Transformer-Enhanced LDA implementation required substantial computational resources, with processing extending over multiple days on a 32 GB RAM system. This computational intensity reflects the method's semantic processing capabilities while raising scalability concerns for larger datasets or real-time applications. The multi-model approach and distribution method would not be appropriate for big data applications.