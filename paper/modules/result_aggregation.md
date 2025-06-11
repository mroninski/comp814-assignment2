## 6.4 Topic Identification and Labeling

The topic identification and labelling process uses a probability-weighted scoring system that solves the fundamental challenges we identified in demographic-specific topic analysis. Our approach recognises that raw topic frequency counts can be misleading when comparing demographics of different sizes, necessitating a normalization strategy that enables fair comparison across groups with varying document volumes.

Probability weighting conversion transforms percentage-based probabilities into numerical weights through division by 100, ensuring that each document's contribution to a topic's aggregate score reflects the confidence level of the topic assignment. This approach acknowledges the inherent uncertainty in probabilistic topic modeling [?] and prevents popular topics from dominating all rankings.

The normalization procedure calculates average topic probabilities per document within each demographic group, dividing aggregate weighted scores by the total number of documents in the group. This normalization strategy addresses the statistical bias that would otherwise favor larger demographic groups and overlabelled topics, ensuring that topic prevalence scores represent the average engagement level per individual rather than absolute topic volume. The mathematical formulation for normalized topic score calculation follows:

> Normalized Score(topic, demographic) = Σ(probability_weight_i) / document_count_demographic

where probability_weight_i represents the converted probability weight for topic occurrence in document i within the specified demographic group.

However, preliminary analysis revealed a critical limitation in using normalized scores alone for demographic comparison. The most popular topics across the entire dataset consistently dominated the rankings for all demographic groups, regardless of the normalization approach. This phenomenon occurs because certain words and terms are closer in the vector space to certain categories. For example, we found generalised abbreviations, such as "lol" being closer to internet categories, which dominate the ranks for that, even after demographic segmentation.

This observation necessitated the development of a demographic specificity scoring methodology that identifies topics exhibiting disproportionate engagement within particular demographic segments relative to their overall popularity. The demographic specificity approach shifts the analytical focus from absolute topic popularity to relative preference patterns, revealing topics that may be moderately popular overall but demonstrate significantly higher engagement within specific demographic groups.

## 7.2 Evaluation Metrics

The evaluation framework incorporates a dual-scoring methodology that addresses both absolute topic popularity and demographic-specific topic preferences. The primary evaluation metric employs the normalized probability scoring described above, which provides comparable measures of topic engagement across demographics of different sizes. This normalization approach ensures that statistical comparisons remain valid regardless of the substantial differences in demographic group sizes documented in the assignment specification.

The secondary evaluation framework implements demographic specificity adjustment, a metric designed to identify topics that exhibit above-average or below-average engagement within specific demographics relative to the general population. The specificity calculation employs a ratio-based approach where demographic-specific topic scores are divided by corresponding overall population scores:

> Demographic Specificity Score(topic, demographic) = Score(topic, demographic) / Score(topic, overall_population)

Specificity scores greater than 1.0 indicate higher-than-average interest in a topic within the demographic, while scores below 1.0 suggest lower engagement relative to the general population. A small epsilon value (1e-12) is added to the denominator to prevent division by zero errors when topics appear exclusively within specific demographics. This demographic specificity methodology draws inspiration from tf-idf weighting principles [?], where term importance is measured relative to document frequency across the entire corpus, but applies this concept to demographic-topic relationships rather than term-document relationships.

The demographic specificity scoring addresses the fundamental challenge identified in preliminary analysis where universally popular topics obscured meaningful demographic differences. By measuring relative engagement rather than absolute popularity, this approach reveals topics that serve as distinguishing characteristics of specific demographic groups, providing actionable insights for the innovation company's product development objectives.

The computational implementation maintains separate aggregation dictionaries for standard normalized scores and demographic specificity scores, enabling comparative analysis between absolute popularity and demographic-specific preferences. Keeping both scores allows us to identify the topics which are globally shared, as well as easily identify the topics that were over-assigned, which could be used to improve the topic modelling process. In the company context, supporting both broad market trend identification and targeted demographic marketing strategies allows for a more nuanced understanding of the overall user behaviour and the needs of the different demographics.