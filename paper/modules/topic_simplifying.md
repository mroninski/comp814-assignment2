### 6.4 Semantic Topic Mapping to Hierarchical Taxonomy

Following the extraction of topic representations from both TF-IDF and Transformer-Enhanced LDA methodologies, a critical challenge emerged in standardizing and interpreting the derived topics within a recognized framework suitable for business intelligence applications. To address this challenge, we implemented a semantic similarity-based mapping system that projects extracted topics onto the Interactive Advertising Bureau (IAB) Content Taxonomy [iabContentTaxonomy], enabling consistent categorization and practical interpretation of demographic-specific content themes. This concept was inspired by [frolov2021using] which utilises the IAB taxonomy to map users to topics that were more closely alligned to their interests.

#### 6.4.1 IAB Content Taxonomy Selection and Acquisition

The IAB Content Taxonomy serves as the foundation for our hierarchical topic mapping framework. This classification system was selected based on its widespread adoption in the advertising industry, which usually requires minimal overlapping in classification[?]. The taxonomy provides a two-tier hierarchical structure comprising major topic categories and their corresponding subtopics, facilitating both broad thematic classification and granular content analysis.

The topic reduction process incorporated an automated taxonomy acquisition system that retrieves the most current version of the IAB Content Taxonomy 3.1 from the official GitHub repository. The taxonomy data is parsed from the original TSV format, where we extract the Tier 1 categories as `major` topics and the Name field as `subtopics`, while filtering out category headers that duplicate their parent categories. Categories lacking subtopic entries are automatically filtered to ensure the resulting taxonomy maintains meaningful hierarchical relationships. This preprocessing yields a structured hashmap mapping major categories to lists of their constituent subtopics, typically encompassing approximately 20-25 major categories with 300-400 total subtopics.

#### 6.4.2 Word Embedding Model Selection and Configuration

The semantic mapping process relies on pre-trained word embedding models to capture semantic relationships between extracted topic terms and taxonomy categories. We selected the `GloVe Twitter 200-dimensional` model as our primary embedding source [pennington2014glove], with the GloVe Wikipedia Gigaword 50-dimensional model serving as a secondary option [pennington2014glove]. The secondary option was used when there were technical issues in using the primary option. The Twitter-trained model was chosen specifically for its relevance to informal, conversational content, similar to our source data.

The GloVe (Global Vectors for Word Representation) approach was found to be a standard method in text mining [?]. Unlike context-dependent embeddings, GloVe vectors maintain consistent representations across different contexts, enabling reliable similarity computations for topic mapping. The Twitter training corpus ensures familiarity with informal language patterns, abbreviations, and colloquialisms prevalent in blog content.

We have handled the vocabulary coverage limitations through a multi-stage word lookup process. For each input term, the system attempts retrieval using the normalized form (lowercase with underscores converted to spaces), the original form, and alternative representations. This meant we were able to maximize vocabulary coverage while maintaining semantic integrity.

#### 6.4.3 Semantic Similarity Computation Framework

The core mapping mechanism employs cosine similarity to quantify semantic relationships between topic representations and taxonomy categories. For individual words, the system directly retrieves pre-trained embeddings where available. For multi-word phrases or compound terms, we implement an averaging-based composition method that computes the mean of constituent word embeddings [?].

The cosine similarity metric is computed using the standard formula:

> similarity(A,B) = (A · B) / (||A|| × ||B||)

where A represents the topic embedding vector and B represents the taxonomy category embedding vector. This metric provides values between -1 and 1, with higher values indicating stronger semantic similarity. The choice of cosine similarity over alternative distance metrics (such as Euclidean distance) reflects its effectiveness in high-dimensional spaces and its invariance to vector magnitude [?].

To optimize computational efficiency, all taxonomy term embeddings are precomputed during system initialization. This precomputation strategy reduces the similarity calculation overhead from O(nm) to O(n), where n represents the number of input topics and m represents the number of taxonomy categories. The precomputed embeddings are stored in memory-efficient arrays, enabling vectorized similarity calculations across entire taxonomies.

#### 6.4.4 Weighted Topic Representation and Quality Integration

The mapping system processes LDA analysis results by constructing weighted topic embeddings that incorporate multiple quality signals. For each topic identified by the LDA analysis, the system extracts the constituent words, their LDA-assigned weights, and coherence scores when available. The word selection process applies a percentile-based filtering approach, retaining words scoring above the 60th percentile of topic weights or ensuring a minimum of three words per topic to maintain representational stability.

The weighted embedding construction employs a dual-weighting scheme that combines LDA topic weights with coherence scores:

> combined_weight = lda_weight × (coherence_score + 0.2)

The addition of 0.2 to coherence scores prevents zero-multiplication scenarios while maintaining the relative importance of coherence in the weighting scheme. This combined weight serves as the input to a weighted averaging calculation:

> topic_embedding = Σ(word_embedding_i × combined_weight_i) / Σ(combined_weight_i)

This approach ensures that highly coherent and topically relevant words exert greater influence on the final topic representation, improving the accuracy of taxonomy mapping while maintaining sensitivity to the LDA model's learned topic structure.

#### 6.4.5 Quality-Weighted Similarity Scoring and Filtering

The final mapping scores undergo quality-based adjustment to reflect the reliability of the underlying topic extraction. Each topic's inherent quality score, derived from the LDA analysis, is transformed into a multiplier ranging from 0.5 to 1.0:

> quality_multiplier = 0.5 + (topic_quality × 0.5)

This transformation ensures that even lower-quality topics maintain some representation in the final mapping while providing enhanced weighting for high-confidence topics. The quality-adjusted similarity scores are computed as:

> final_similarity = cosine_similarity × quality_multiplier

A minimum similarity threshold of 0.50 is applied to filter out weak semantic connections, ensuring that only meaningful topic-taxonomy associations are retained in the final results. This threshold was empirically determined to balance coverage with precision, reducing noise while maintaining sufficient mapping density for meaningful analysis.

The system outputs the top 25 topic-taxonomy mappings per demographic group, formatted as "major_category:subtopic" pairs with percentage similarity scores. Duplicate mappings are resolved by retaining the highest-scoring association, ensuring that each taxonomy category appears at most once in the final results while preserving the strongest semantic connections identified by the analysis.

#### 6.4.6 Integration with Demographic-Specific Analysis

The taxonomy mapping system operates independently on the topic extraction results from each demographic segment, enabling comparative analysis of topic preferences across different user groups. By applying consistent mapping procedures to all demographic categories (males, females, age brackets, students, and the general population), the system facilitates direct comparison of topic prevalence and semantic similarity patterns across different user segments.

This standardized mapping approach addresses the challenge of comparing topics extracted using different methodologies (TF-IDF versus Transformer-Enhanced LDA) by projecting both sets of results onto the common IAB taxonomy framework. The resulting mapped topics enable business stakeholders to identify demographic-specific content preferences using industry-standard category definitions, supporting practical decision-making for product development and marketing strategies.