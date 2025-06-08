# COMP814 Text Mining Assignment Structure

## 1. Title
**Title:** "Comparative Analysis of Topic Extraction Methods for Early 2000s Blog Content Mining"

## 2. Abstract (0.25 pages)
- Brief overview of the blog mining task
- Mention of two topic extraction strategies: TF-IDF and Transformer-Enhanced LDA
- Key findings about the top topics identified in the dataset
- Performance discussion due to the scalability of the process, as it would not scale

## 3. Contributions (0.25 pages)
**Section detailing individual contributions from each team member**
- Data preprocessing and cleaning pipeline development (Pedro)
- Implementation of topic extraction methods (Pedro + Kent)
- Demographic segmentation and analysis (Pedro + Kent)
- Results evaluation and documentation (Kent)

## 4. Introduction (1 page)
### 4.1 Background and Motivation
- Importance of understanding demographic-specific content trends
- Value for product/service innovation based on blog analysis
- Overview of 19,320 blog dataset (2001-2004) (`schler2006effects` paper on references.bib)

### 4.2 Research Objectives
- Innovation company context and business value
  - This would be focused on the assumptions of the company, and the value of the analysis
  - Assuming we are working to identify probable marketing opportunities, we would want to identify the most popular topics for each demographic
- Primary goal: Extract two most popular topics per demographic
- Secondary goals: Compare extraction methods, understand demographic differences

### 4.3 Paper Organization
- Brief roadmap of remaining sections
- Focus on the change from the assumed expectation to the reality of the task

## 5. Literature Review (1.5 pages)
### 5.1 Topic Modeling Approaches
- Traditional methods: LSA, pLSA, basic LDA
- Modern enhancements: Transformer-based models, semantic embeddings
- Previous work on demographic-based content analysis

### 5.2 TF-IDF for Topic Extraction
- Theoretical foundation and applications
- Strengths in identifying distinctive terms
- Limitations in capturing semantic relationships

### 5.3 LDA and Neural Enhancements
- Evolution from basic LDA to transformer-enhanced approaches
- Benefits of combining probabilistic models with embeddings
- Recent applications in short-text analysis

## 6. Methodology (3 pages)
### 6.1 Data Description and Preprocessing
- Dataset structure: XML files with metadata in filenames
- Demographic distribution table (already provided)
- Cleaning pipeline: HTML removal, tokenization, lemmatization
- Handling of non-ASCII characters and noise

### 6.2 Demographic Segmentation Strategy
- Metadata extraction from filenames
- Creation of five demographic groups
- Data validation and consistency checks

### 6.3 Topic Extraction Method 1: TF-IDF Approach
- Document segmentation using semantic boundaries
- Phrase detection with Gensim
- TF-IDF vectorization parameters (max_features=1000, ngram_range=(1,3))
- Topic quality scoring and filtering

### 6.4 Topic Extraction Method 2: Transformer-Enhanced LDA
- Semantic document creation using spaCy
- Integration of Sentence-BERT embeddings (all-mpnet-base-v2)
- Adaptive topic number optimization via silhouette analysis
- Semantic re-ranking of topic words using centroid similarity

### 6.5 Topic Identification and Labeling
- Common post-processing pipeline for both methods
- Generation of topic labels from top words
- Extraction of clauses containing dominant topics
- Quality-based filtering (threshold determination)

## 7. Experimental Setup (1 page)
### 7.1 Implementation Environment
- Google Colab setup with drive mounting
- Required libraries: spaCy, Gensim, scikit-learn, sentence-transformers
- Computational considerations for 19,320 files

### 7.2 Evaluation Metrics
- Topic coherence measures
- Manual evaluation criteria
- Inter-annotator agreement for topic relevance

### 7.3 Parameter Optimization
- Hyperparameter tuning for both methods
- Cross-validation strategy for demographic subsets

## 8. Results and Analysis (2.5 pages)
### 8.1 Overall Topic Distribution
- Top topics across entire dataset
- Topic frequency and quality scores
- Word clouds or visualization of major themes

### 8.2 Demographic-Specific Results
#### 8.2.1 Gender-Based Analysis
- Male vs. Female topic differences
- Statistical significance testing

#### 8.2.2 Age-Based Analysis
- Under 20 vs. Over 20 comparisons
- Generational topic preferences

#### 8.2.3 Student Population
- Unique topics for student demographic
- Comparison with general population

### 8.3 Method Comparison
- TF-IDF vs. Transformer-LDA performance
- Topic quality and coherence scores
- Computational efficiency analysis
- Agreement between methods on top topics
  - The top topics were always the same, so we had to identify the most relevant topics for each demographic.
    - These would be the topics that are have more posts on that demographic compared to the rest of the demographics.

### 8.4 Example Topic Clauses
- Representative clauses for each demographic's top 2 topics
- Context and interpretation

## 9. Discussion (1 page)
### 9.1 Key Findings
- Most significant demographic differences
- Unexpected topic discoveries
- Business implications for innovation

### 9.2 Method Strengths and Limitations
- When TF-IDF excels vs. Transformer-LDA
- Trade-offs between interpretability and semantic depth
- Scalability considerations
  - Transformer-LDA is incredibly heavy on the computational resources, and so it would not be feasible to run on a larger dataset.

### 9.3 Challenges Encountered
- Data quality issues (HTML artifacts, encoding)
  - The cleanup required
  - Computational considerations due to the model sizes used for the LDA-Transformer method
- Temporal aspects (2001-2004 content relevance)
  - Language used (Users were more likely to use slang and abbreviations in the early 2000s)
  - As the internet grew, more data was available, and so language models are not as effective to specific lingo to the beginning of the internet
- Short text and sparse data handling
  - Lots of shorter posts which forced us to process the blogs in full, instead of per post as it would be more specific and could produce more relevant results based on time of posting.

## 10. Conclusion and Future Work (0.75 pages)
### 10.1 Summary of Contributions
- Successfully extracted demographic-specific topics
- Comprehensive comparison of extraction methods
- Practical insights for product innovation

### 10.2 Recommendations
- Preferred method based on use case
- Implementation guidelines for production

### 10.3 Future Improvements
- Real-time topic tracking
- Integration with modern blog platforms
- Deep learning approaches (BERT fine-tuning)
- Temporal analysis of topic evolution

## 11. References
- IEEE format citations
- Mix of foundational papers and recent advances

## 12. Appendix
### A. Code Repository
- Google Colab notebook link with full implementation
- Instructions for data placement in Google Drive
- Requirements and setup guide

---

## Key Considerations:
1. **Balance detail with page limits** - Methodology gets most space as per requirements
2. **Focus on comparison** - Central theme is comparing two methods
3. **Emphasize practical insights** - Remember the innovation company context and the assumptions made about the task for each section of the analysis
4. **Include all required elements** - Especially the clause extraction examples
5. **Visual elements** - Reserve space for tables, figures in Results section