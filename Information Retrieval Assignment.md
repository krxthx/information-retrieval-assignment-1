# Information Retrieval (S2-25_AIMLZG537)

## Assignment #1

### Query Clustering for Intent Discovery in Web Search

---

## 1. Problem Statement

Modern web search systems must understand user intent behind queries. Similar queries often reflect shared intent (e.g., "weather Goa", "temperature in Goa").

Your task is to:

- Represent queries in vector space
- Cluster them to discover latent intents
- Analyse how clustering can improve retrieval or suggestion systems

This assignment emphasizes query-side intelligence rather than document retrieval alone.

---

## 2. Objectives

1. Represent queries using appropriate vector representations
2. Apply clustering algorithms
3. Interpret discovered intent groups
4. Evaluate clustering quality
5. Explore cross-lingual query clustering

---

## 3. Dataset Requirements

You may use any public query datasets. It's recommended to use at least 10% multilingual queries.

---

## 4. Technical Requirements

### 4.1 Query Representation

Implement at least two:

- TF-IDF
- N-gram vectors
- Embedding-based representations

### 4.2 Clustering Algorithms (Choose at least two)

- K-means
- Hierarchical clustering
- DBSCAN

You must:

- Experiment with hyperparameters
- Analyse cluster stability

### 4.3 Cross-Lingual Query Clustering (Optional)

Support clustering across languages using:

- Translation-based normalization

**OR**

- Shared multilingual embeddings

### 4.4 Text Mining Component

- Frequent intent pattern extraction
- Cluster labelling using top terms

---

## 5. Evaluation

- Silhouette score

Additionally:

- Show how clustering improves query suggestions or retrieval grouping

---

## 6. Deliverables

1. A zip file containing source code, and assignment report strictly in PDF format including Visualization of clusters.
2. Naming convention for assignment, `Group#-<No>-Assignment-1.pdf`
3. Whatever assumptions you are making mention it clearly and why
4. Strictly adhere to assignment guidelines, and there will be no extension under any circumstances
5. Make sure to submit assignment as PDF file only
6. If zip file size including code is more than 10 MB, share source code link in pdf file.