# Literature Review: Topic Modeling Approaches

## 5. Literature Review

### 5.1 Topic Modeling Approaches

Latent Semantic Analysis (LSA) was introduced by Deerwester et al. [deerwester_1990], which employs Singular Value Decomposition to analyze relationships between documents and terms through implicit higher-order semantic structure.  By constructing a term-by-document matrix and decomposes it into orthogonal factors, representing documents as vectors of factor weights,we can address the limitations of exact keyword matching by capturing semantic relationships and handling synonymy and polysemy issues in information retrieval.

A probabilistic frameworks emerged with Hofmann's Probabilistic Latent Semantic Analysis (pLSA) [hofmann_1999] [hofmann_2001], which introduced statistical foundations for topic modeling through a generative latent class model. Unlike LSA's linear algebra approach, pLSA models the probability of word-document co-occurrences as a mixture of conditionally independent multinomial distributions, where:

> P(d,w) = P(d)P(w|d) and P(w|d) = Σ P(w|z)P(z|d) over latent topics z

Parameters are learned using the Expectation-Maximization algorithm, specifically a temperature-controlled version to avoid overfitting.

Latent Dirichlet Allocation, proposed by Blei et al. [blei_2003], established a three-level hierarchical Bayesian model representing documents as finite mixtures over latent topics. LDA addresses pLSA's overfitting and new document problems through Dirichlet priors, creating a fully generative model. The approach assumes a Dirichlet prior distribution over per-document topic distributions, with inference performed using variational methods and empirical Bayes parameter estimation.

Modern enhancements have integrated transformer-based models with traditional probabilistic frameworks. The Embedded Topic Model (ETM) [dieng_2020] works by modeling words through categorical distributions whose natural parameters are inner products between word embeddings and topic embeddings. This approach places both words and topics in the same embedding space, achieving better topic coherence and handling large vocabularies more effectively than traditional LDA. Contextualized Topic Models (CTM) [bianchi_2021] further advance this direction by combining BERT contextual embeddings with Neural ProdLDA, producing more meaningful and coherent topics than traditional bag-of-words approaches.

### 5.2 LDA-Transformer

There have been substantial improvements in topic coherence and semantic understanding through the usage of contextual embeddings. The Transformer-Representation Neural Topic Model (TNTM) [reuter_2024] shows this evolution by combining transformer embedding spaces with fully probabilistic modeling, similar to LDA, achieving high quality topic diversity while maintaining embedding coherence.

BERTopic [grootendorst_2022] introduces a clustering-based approach using transformer embeddings with a novel class-based TF-IDF (c-TF-IDF) procedure for topic representation extraction. The method employs a highly modular architecture that allows component swapping. Comparative studies [egger_2022] demonstrate BERTopic's superior performance over traditional LDA for social media analysis, particularly on Twitter data where short text sparsity challenges traditional approaches.

The benefits of combining probabilistic models with embeddings are particularly evident in addressing fundamental limitations of bag-of-words representations. The Contextualized Word Topic Model (CWTM) [fang_2024] integrates contextualized BERT embeddings without relying on bag-of-words assumptions, effectively handling out-of-vocabulary words crucial for social media applications.

Contextual-Top2Vec [angelov_2024] creates hierarchical topics and finds topic spans within documents, labeling topics with phrases rather than individual words. The approach outperforms current state-of-the-art models on comprehensive topic model evaluation metrics, particularly excelling in social media contexts where traditional LDA struggles with sparse, informal text. The Biterm Topic Model (BTM) [yan_2013] [cheng_2014] addresses short-text challenges by directly modeling word co-occurrence patterns globally rather than at the document level, achieving significantly better performance than LDA on microblog content.

These methods habe been manually evaluated. Eklund and Forsman [eklund_2022] demonstrate that clustering language model embeddings with BERT achieves coherent topic creation in production systems, addressing real-world requirements for scalable topic modeling. The combination of transformer representations with probabilistic frameworks has proven particularly effective for demographic analysis, multilingual applications, and dynamic topic tracking in streaming social media data.