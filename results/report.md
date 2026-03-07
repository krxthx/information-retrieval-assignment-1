# Assignment 1 Report Draft

## Problem Statement

This project clusters web search queries to discover latent user intents and to show how query-side clustering can improve suggestion quality and retrieval grouping.

## Dataset

- Total queries: 95
- Multilingual queries: 15.8%
- Domain: Goa travel-related search queries spanning weather, hotels, restaurants, tourism, flights, local transport, packages, and multilingual variants.

## Methodology

- Preprocessing: Unicode normalization, lowercasing, punctuation cleanup, translation-based normalization for Hindi and Tamil queries, and intent-tag heuristics for downstream evaluation.
- Query representations: word TF-IDF, character n-gram TF-IDF, and a dense embedding representation. The pipeline uses a local sentence-transformer when cached, and otherwise falls back to an offline LSA embedding so the experiment remains reproducible.
- Clustering algorithms: K-means, Agglomerative hierarchical clustering, and DBSCAN.
- Evaluation: silhouette score, Davies-Bouldin index, Calinski-Harabasz score, cluster purity against heuristic intent tags, bootstrap-style stability using adjusted Rand index, and suggestion precision at 3.

## Best Configuration

- Best run: embedding [sentence_transformer] + kmeans (n_clusters=8)
- Silhouette score: 0.3577
- Stability: 0.7660
- Cluster purity: 0.8371
- Suggestion lift over baseline precision@3: -0.0035

## Cluster Interpretation

### Cluster 0: nightlife
- Size: 1
- Top terms: nightlife
- Frequent patterns: nightlife
- Intent mix: tourism=1
- Representative queries: goa nightlife

### Cluster 1: flight / flight booking
- Size: 12
- Top terms: flight, flight booking, flight deal, delhi, flight mumbai
- Frequent patterns: flight, ticket, mumbai, flight ticket, flight mumbai
- Intent mix: flight=10, package=1, hotel=1
- Representative queries: flights to goa | गोवा फ्लाइट | கோவா விமானம்

### Cluster 2: weather / temperature
- Size: 23
- Top terms: weather, temperature, climate, humidity, december
- Frequent patterns: weather, temperature, climate, december, forecast
- Intent mix: weather=23
- Representative queries: கோவா வானிலை | गोवा मौसम | weather goa

### Cluster 3: travel / package
- Size: 23
- Top terms: travel, package, travel package, holiday, things
- Frequent patterns: travel, package, travel package, holiday, visit
- Intent mix: tourism=13, package=8, other=1, transport=1
- Representative queries: goa tourism | गोवा पर्यटन | கோவா சுற்றுலா

### Cluster 4: restaurant / food
- Size: 10
- Top terms: restaurant, food, places, cafe, food places
- Frequent patterns: restaurant, food, places, vegetarian restaurant, vegetarian
- Intent mix: restaurant=10
- Representative queries: restaurants in goa | goa food places | goa dinner places

### Cluster 5: rental / car
- Size: 9
- Top terms: rental, car, car rental, taxi, scooter
- Frequent patterns: rental, car, taxi, car rental, scooter
- Intent mix: transport=8, hotel=1
- Representative queries: goa rental vehicles | goa car hire | car rental goa

### Cluster 6: beach / resort
- Size: 6
- Top terms: beach, resort, beach hotel, beach holiday, beach resort
- Frequent patterns: beach, resort, hotel, holiday, family resort
- Intent mix: hotel=3, tourism=2, package=1
- Representative queries: goa beach resorts | गोवा बीच | best beaches in goa

### Cluster 7: hotel / hotel booking
- Size: 11
- Top terms: hotel, hotel booking, booking, hotel deal, star hotel
- Frequent patterns: hotel, hotel booking, booking, star hotel, star
- Intent mix: hotel=10, package=1
- Representative queries: गोवा होटल | கோவா ஹோட்டல் | luxury hotels goa

## Cross-Lingual Clustering

Hindi and Tamil queries are normalized into the same English intent space before vectorization. This translation-based normalization lets multilingual queries join the same clusters as their English counterparts instead of forming isolated script-specific groups.

## Query Suggestion Improvement

- Best clustering run for overall balance: embedding [sentence_transformer] + kmeans (n_clusters=8)
- Best downstream suggestion run: tfidf_char [tfidf_char] + agglomerative (n_clusters=10, metric=cosine, linkage=average)
- Baseline precision@3: 0.8211
- Cluster-constrained precision@3: 0.8702
- Absolute lift in precision@3: 0.0491

## Assumptions

- The provided dataset is a representative sample of Goa travel-related intents.
- Intent tags used in the downstream evaluation are heuristic labels derived from normalized query terms.
- Translation-based normalization is sufficient for the current multilingual subset.

## Limitations

- The dataset is small and synthetic, so conclusions should be framed as a controlled experiment rather than a production benchmark.
- Heuristic intent tags are useful for analysis, but they are not human-annotated ground truth.

## Generated Artifacts

- `results/final_query_clusters.csv`
- `results/cluster_summary.md`
- `results/report.md`
- `results/figures/experiment_scores.png`
- `results/figures/best_clusters.png`
- `results/figures/cluster_sizes.png`

## Top Experiment Configurations

```text
representation representation_backend  feature_count     algorithm                                 param_summary  stability  cluster_count  noise_points  noise_ratio  assigned_ratio  largest_cluster_share  mean_cluster_size  tiny_cluster_ratio  silhouette  davies_bouldin  calinski_harabasz  cluster_purity  baseline_precision_at_3  cluster_precision_at_3  suggestion_lift  cluster_suggestion_coverage  selection_score
     embedding   sentence_transformer            384        kmeans                                  n_clusters=8     0.7660            8.0           0.0          0.0             1.0                 0.2421            11.8750              0.0105      0.3577          1.4626            11.4081          0.8371                   0.8667                  0.8632          -0.0035                       0.9895           1.2802
     embedding   sentence_transformer            384 agglomerative  n_clusters=8, metric=cosine, linkage=average     0.9924            8.0           0.0          0.0             1.0                 0.3579            11.8750              0.0421      0.3183          1.3673             9.2222          0.8636                   0.8667                  0.8737           0.0070                       0.9789           1.2670
     embedding   sentence_transformer            384 agglomerative n_clusters=10, metric=cosine, linkage=average     0.8859           10.0           0.0          0.0             1.0                 0.2737             9.5000              0.0526      0.3289          1.2492             8.8445          0.9060                   0.8667                  0.8702           0.0035                       0.9684           1.2577
     embedding   sentence_transformer            384 agglomerative  n_clusters=9, metric=cosine, linkage=average     0.9743            9.0           0.0          0.0             1.0                 0.3579            10.5556              0.0526      0.3101          1.2751             8.3721          0.8834                   0.8667                  0.8702           0.0035                       0.9684           1.2403
     embedding   sentence_transformer            384        kmeans                                 n_clusters=10     0.7427           10.0           0.0          0.0             1.0                 0.1579             9.5000              0.0105      0.3151          1.5629            10.8783          0.8998                   0.8667                  0.8632          -0.0035                       0.9895           1.2162
     embedding   sentence_transformer            384        kmeans                                  n_clusters=7     0.7157            7.0           0.0          0.0             1.0                 0.2947            13.5714              0.0105      0.3182          1.6984            10.9352          0.8821                   0.8667                  0.8526          -0.0140                       0.9895           1.2028
     embedding   sentence_transformer            384        kmeans                                  n_clusters=9     0.7685            9.0           0.0          0.0             1.0                 0.2421            10.5556              0.0105      0.2882          1.6747            10.5908          0.8552                   0.8667                  0.8632          -0.0035                       0.9895           1.1924
     embedding   sentence_transformer            384 agglomerative  n_clusters=7, metric=cosine, linkage=average     0.8861            7.0           0.0          0.0             1.0                 0.3579            13.5714              0.0421      0.2899          1.3913             8.8574          0.8072                   0.8667                  0.8632          -0.0035                       0.9789           1.1657
     embedding   sentence_transformer            384        kmeans                                  n_clusters=6     0.5901            6.0           0.0          0.0             1.0                 0.2947            15.8333              0.0000      0.3219          1.8746            12.5589          0.8473                   0.8667                  0.8526          -0.0140                       1.0000           1.1454
     embedding   sentence_transformer            384 agglomerative  n_clusters=6, metric=cosine, linkage=average     0.8873            6.0           0.0          0.0             1.0                 0.3579            15.8333              0.0211      0.2822          1.4724             9.9099          0.7820                   0.8667                  0.8632          -0.0035                       0.9789           1.1396
    tfidf_char             tfidf_char           1008 agglomerative n_clusters=10, metric=cosine, linkage=average     0.7659           10.0           0.0          0.0             1.0                 0.3579             9.5000              0.0316      0.2672          1.9503             4.6391          0.8803                   0.8211                  0.8702           0.0491                       0.9895           1.1278
    tfidf_char             tfidf_char           1008 agglomerative  n_clusters=9, metric=cosine, linkage=average     0.7269            9.0           0.0          0.0             1.0                 0.3579            10.5556              0.0316      0.2502          1.9797             4.6223          0.8669                   0.8211                  0.8702           0.0491                       0.9895           1.1068
    tfidf_char             tfidf_char           1008 agglomerative  n_clusters=8, metric=cosine, linkage=average     0.7321            8.0           0.0          0.0             1.0                 0.4316            11.8750              0.0316      0.2406          1.9340             4.7153          0.8766                   0.8211                  0.8561           0.0351                       0.9895           1.0872
    tfidf_word             tfidf_word            116        kmeans                                  n_clusters=9     0.5451            9.0           0.0          0.0             1.0                 0.3789            10.5556              0.0105      0.2905          1.9611             5.5315          0.8169                   0.8211                  0.7930          -0.0281                       0.9895           1.0646
    tfidf_char             tfidf_char           1008 agglomerative  n_clusters=7, metric=cosine, linkage=average     0.7554            7.0           0.0          0.0             1.0                 0.4316            13.5714              0.0105      0.2268          2.0301             4.9290          0.8493                   0.8211                  0.8421           0.0211                       0.9895           1.0572
```
