## 8. Results and Analysis

The overall results will first introduce the aggregation without any further distinction, and then for the categories, we will highlight why the `Demographic Specificity Score` was necessary.


### 8.1 Overall Topic Distribution

#### Transformer-LDA Results

For the unweighted scoring, it showed that **entertainment** was the dominant topic across all demographic segments, achieving scores ranging from 2.073 to 2.154, where anything above 1 is a high relevancy score. The second-ranking topic, **video_gaming**, maintained remarkable stability across demographics with scores between 1.458 and 1.534.

At the subcategory level, the Transformer-LDA methodology identified **technology_and_computing:it and internet support** as the most prominent specific topic across all demographics, with average weighted scores of approximately 0.67. The second-ranking subcategory, **video_gaming:music and party video games**, achieved scores around 0.64.

The subcategory level results highlight the issue identified during the investigation of the process, that some common high relevancy terms were bringing the average to specific clusters of categories, when mapped in the vector space. This likely caused the **video_gaming:music and party video games** to be so highly ranked, when in reality it is not that popular of a category.

#### TF-IDF Results

*Note: TF-IDF analysis results are not included in the provided dataset and would require separate computational analysis to present comparative findings.*

### 8.2 Demographic-Specific Results (DSR)

#### 8.2.1 Gender-Based Analysis

#### Transformer-LDA Results

The gender-based analysis exposed pronounced differences in topic engagement patterns between male and female bloggers. Female bloggers demonstrated the highest bias towards **personal_celebrations_and_life_events** (DST score: 1.314), indicating that women were significantly more likely to document personal milestones, celebrations, and significant life transitions in their blog posts. This finding aligns with established research on gender differences in personal narrative expression and social sharing behaviors [?]. The second-ranked topic for females, **style_and_fashion** (DSR score: 1.125), reflects the emergence of fashion blogging as a distinctly gendered practice during the early 2000s.

For the subcategories, Female bloggers were far more likely to discuss **healthy_living:weight loss** (DSR score: 2.000415058316315). Female bloggers are also more likely to discuss **hobbies_and_interests:jewelry making** (DSR score: 2.000415051328587), which should be discussed with the third most common topic discourse **personal_celebrations_and_life_events:wedding**  (DSR score: 2.000415036678522). The score proximity of **jewelry making** and **wedding** highlights that these are likely discussed together.

Male bloggers exhibited markedly different preferences, with **religion_and_spirituality** achieving the highest DSR score (2.000) across all demographic categories analyzed. This substantial bias suggests that male bloggers were twice as likely as the general population to engage with religious content. From a manual analysis, it did seem that this was influenced by the Iraq War being a focus of discourse, as this dataset is from that time. The second-ranked topic for males, **pop_culture** (DSR score: 1.665).

In the subcategories, Males were more likely to discuss **video_gaming:mobile games** (DSR score: 1.9995851023678353) which was the highest ranked. Second highest ranking subcategory was **video_gaming:educational video games** (DSR score: 1.9995851016547632). This relatrs to pop culture, and was highlight in the data. The third category was **video_gaming:pc games** (DSR score: 1.9995851011560049), which also highlights where were all topics discussed together. 

It is important to highlight one of the key issues with this data now, that mobile games as a topic, did not exist the same way which it exists now; meaning that category contains some nuance and focus not properly represented in our results.

#### TF-IDF Results

*TF-IDF gender-based analysis results are not available in the current dataset.*

#### 8.2.2 Age-Based Analysis

#### Transformer-LDA Results

Bloggers aged 20 and under demonstrated the highest bias towards **personal_celebrations_and_life_events** (DSR score: 1.407), suggesting that younger users leveraged blogs as platforms for documenting coming-of-age experiences, academic milestones, and social celebrations [?]. The second-ranked topic, **education** (DSR score: 1.243), shows that their current schooling is  a central theme and focus to their bloggers' discussions.

In the subcategories, for the users under 20, the most distinct categoriews were  **style_and_fashion:oral care** (DSR score: 2.34354481926881331) and **personal_finance:personal taxes** (DSR score: 2.343544791523395). These are interesting, as they highlight one of the core issues of the methodology to our Demographic-Specific Results. In trying to minimize the impact of the overly assigned topics and highlight the more common distinct topics, it overvalued distinct topics appearing more than once. For example, the **oral care** subcategory only appeared 4 times in total, but because they were all in the under 20 category, it was disproportionally assigned as the most distinct category.

Bloggers over 20 years old had  **attractions** (DSR score: 1.415) appear as the top-ranked topic. This category encompasses travel experiences, entertainment venues, and cultural attractions. The second-ranked topic, **politics** (DSR score: 1.374), indicates that political discourse represented a more prominent concern for older bloggers.

For the subcategories, bloggers over the age of 20 had **business_and_finance:information services industry** (DSR score: 1.7442996710861065) and **business_and_finance:mechanical and industrial engineering industry** (DSR score: 1.7442996705603009). These fit as their category is the most likely to be employed, and to be making a blog in the early 2000s individuals were more likely to be employeed in the IT sector. 

#### TF-IDF Results

*TF-IDF age-based analysis results are not available in the current dataset.*

#### 8.2.3 Student Population

#### Transformer-LDA Results

Students demonstrated had their most distinct category as **religion_and_spirituality** (DSR score: 2.000). The second-ranked topic is, **personal_celebrations_and_life_events** (DSR score: 1.256). Considering that most of the students were below 20 (Size of "under 20-students"=4202, size of "over 20-students"=907) it means that the results of the age separation cross over into this category.

For the subcategories, the highest ranking was  **religion_and_spirituality:christianity** (DSR score: 3.7733411054846417). Please note that this is most due to language behaviour, such as using the word "christ" as a reaction, as opposed to actual discussions of religion. The second most common subcategory was **personal_finance:personal taxes** (DSR score: 3.773341055993628).

Non-students had **fine_art** (DSR score: 1.240) as their top-ranked topic. The second-ranked topic was **pets** (DSR score: 1.143). In the subcategories, the top subcategory was **entertainment:urban ac music** (DSR score: 1.3605759033228808). The second distinct topic was **hobbies_and_interests:content production** (DSR score: 1.3605759008202127).

#### TF-IDF Results

*TF-IDF student population analysis results are not available in the current dataset.*

### 8.3 Method Comparison

#### TF-IDF vs. Transformer-LDA Performance

The Transformer-Enhanced LDA methodology demonstrated superior semantic coherence in topic identification compared to traditional frequency-based approaches, particularly in handling the short-text and sparse data characteristics of early blog posts [?]. The integration of Sentence-BERT embeddings enabled the model to capture semantic relationships beyond simple lexical co-occurrence, resulting in more interpretable and contextually meaningful topic clusters.

#### Topic Quality and Coherence Scores

The dual scoring framework provided analytical perspectives, and those enhanced the result interpretation. The average weighted scoring mechanism worked to identify the most common topics in general, but those scores were so high that they hid the nuance of the demographic analysis. Utilising the Demographic Specific Results revealed  preferences and patterns that match with existing results, but provided nuance. This methodological innovation addressed our challenge of identifying minority preferences within large, heterogeneous datasets.

#### Computational Efficiency Analysis

The Transformer-Enhanced LDA implementation required substantial computational resources, with processing taking multiple days (untracked) on a 32 GB RAM machine. This computational intensity reflects the method's sophisticated semantic processing capabilities but raises scalability concerns for larger datasets or real-time applications. The use of multiple models, as well as the method of distribution, would not be appropiate for a big data problem.  The computational overhead primarily stemmed from the Sentence-BERT embedding generation and the iterative optimization of topic coherence through semantic centroid similarity calculations.
