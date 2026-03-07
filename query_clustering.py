import pandas as pd
import numpy as np
import re
import warnings
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available, using TF-IDF only")

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = "output"

def load_public_dataset():
    """Load the QueriesByCountry dataset for clustering experiments."""
    
    print("   Loading QueriesByCountry dataset...")
    
    df = pd.read_csv('data/QueriesByCountry_2021-04-01_2021-04-30.tsv', sep='\t')
    
    queries = df['Query'].tolist()
    countries = df['Country'].tolist()
    
    query_country = list(zip(queries, countries))
    query_country = list(set(query_country))
    queries = [q for q, c in query_country]
    countries = [c for q, c in query_country]
    
    np.random.seed(42)
    indices = np.random.choice(len(queries), min(2000, len(queries)), replace=False)
    queries = [queries[i] for i in indices]
    countries = [countries[i] for i in indices]
    
    intents = infer_intent_from_query(queries)
    
    print(f"   Loaded {len(queries)} unique queries")
    print(f"   Countries: {len(set(countries))}")
    
    return queries, intents, countries

def infer_intent_from_query(queries):
    """Infer intent categories from query text."""
    intent_keywords = {
        'health_covid': ['covid', 'coronavirus', 'vaccine', 'vaccination', 'vaccin', 'pfizer', 'moderna', 
                        'sintomas', 'symptoms', 'cases', 'deaths', 'cases', 'hospital', 'icu'],
        'health_general': ['symptoms', 'treatment', 'health', 'doctor', 'medicine', 'pharmacy', 'hospital'],
        'travel': ['flight', 'hotel', 'travel', 'vacation', 'booking', 'book', 'train', 'ticket'],
        'shopping': ['buy', 'shop', 'price', 'order', 'online', 'store', 'amazon', 'ebay'],
        'food': ['recipe', 'food', 'restaurant', 'cook', 'eating', 'menu', 'delivery'],
        'technology': ['update', 'install', 'download', 'app', 'software', 'tech', 'computer', 'phone'],
        'news': ['news', 'update', 'latest', 'today', 'breaking', 'report'],
        'government': ['government', 'gov', 'official', 'certificate', 'license', 'permit'],
        'entertainment': ['movie', 'film', 'music', 'song', 'game', 'play', 'netflix', 'youtube'],
        'weather': ['weather', 'forecast', 'temperature', 'rain', 'snow', 'climate']
    }
    
    intents = []
    for query in queries:
        query_lower = query.lower()
        matched = False
        for intent, keywords in intent_keywords.items():
            if any(kw in query_lower for kw in keywords):
                intents.append(intent)
                matched = True
                break
        if not matched:
            intents.append('other')
    
    return intents

class QueryPreprocessor:
    def __init__(self):
        self.stopwords = {
            'english': {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are', 'was', 'were', 
                       'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                       'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
                       'used', 'how', 'what', 'who', 'whom', 'which', 'where', 'when', 'why', 'all', 'each',
                       'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                       'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'don', 'now',
                       'and', 'but', 'or', 'as', 'if', 'then', 'because', 'while', 'about', 'into', 'through',
                       'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down', 'out', 'off',
                       'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'this', 'that'},
            'german': {'der', 'die', 'das', 'und', 'oder', 'aber', 'nicht', 'ein', 'eine', 'einer', 'einem',
                      'einen', 'den', 'dem', 'des', 'zu', 'zur', 'zum', 'von', 'mit', 'auf', 'in', 'an',
                      'als', 'auch', 'es', 'ist', 'sind', 'war', 'waren', 'hat', 'haben', 'wird', 'werden',
                      'kann', 'können', 'konnte', 'konnten', 'muss', 'müssen', 'musste', 'mussten', 'soll',
                      'sollen', 'sollte', 'sollten', 'will', 'wollen', 'wollte', 'wollten', 'darf', 'dürfen',
                      'durfte', 'durften', 'mag', 'mögen', 'mochte', 'mochten', 'ich', 'du', 'er', 'sie',
                      'es', 'wir', 'ihr', 'mich', 'dich', 'sich', 'uns', 'euch', 'mein', 'dein', 'sein',
                      'ihr', 'unser', 'euer', 'was', 'wer', 'wie', 'wo', 'wann', 'warum', 'welche', 'welcher'},
            'french': {'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'au', 'aux', 'et', 'ou', 'mais',
                      'donc', 'car', 'ne', 'pas', 'plus', 'moins', 'très', 'bien', 'mal', 'je', 'tu', 'il',
                      'elle', 'nous', 'vous', 'ils', 'elles', 'me', 'te', 'se', 'lui', 'leur', 'moi', 'toi',
                      'soi', 'nous', 'vous', 'eux', 'elles', 'mon', 'ton', 'son', 'ma', 'ta', 'sa', 'mes',
                      'tes', 'ses', 'notre', 'votre', 'leur', 'nos', 'vos', 'leurs', 'ce', 'cet', 'cette',
                      'ces', 'qui', 'que', 'quoi', 'dont', 'où', 'quand', 'comment', 'pourquoi', 'être',
                      'avoir', 'faire', 'pouvoir', 'vouloir', 'devoir', 'savoir', 'être', 'est', 'sont',
                      'était', 'étaient', 'ai', 'as', 'avons', 'avez', 'ont', 'avait', 'avaient'},
            'spanish': {'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'de', 'del', 'al', 'a', 'en',
                       'y', 'o', 'u', 'pero', 'porque', 'que', 'cual', 'cuales', 'quien', 'quienes', 'donde',
                       'cuando', 'como', 'por', 'para', 'sin', 'sobre', 'entre', 'yo', 'tu', 'el', 'ella',
                       'nosotros', 'vosotros', 'ellos', 'ellas', 'me', 'te', 'se', 'nos', 'os', 'le', 'les',
                       'mi', 'tu', 'su', 'mis', 'tus', 'sus', 'nuestro', 'vuestro', 'este', 'esta', 'estos',
                       'estas', 'ese', 'esa', 'esos', 'esas', 'aquel', 'aquella', 'aquellos', 'aquellas',
                       'ser', 'estar', 'tener', 'hacer', 'poder', 'querer', 'saber', 'decir', 'es', 'son',
                       'era', 'eran', 'tiene', 'tienen', 'hace', 'hacen'}
        }
        self.all_stopwords = set()
        for lang_sw in self.stopwords.values():
            self.all_stopwords.update(lang_sw)
    
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.all_stopwords and len(t) > 1]
        return ' '.join(tokens)

class QueryVectorizer:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.tfidf_vectorizer = None
        self.ngram_vectorizer = None
        self.embedding_model = None
        self.pca = None
        
    def create_tfidf_vectors(self, queries, ngram_range=(1, 2), max_features=500):
        processed_queries = [self.preprocessor.preprocess(q) for q in queries]
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=1,
            max_df=0.95
        )
        return self.tfidf_vectorizer.fit_transform(processed_queries)
    
    def create_ngram_vectors(self, queries, n=3, max_features=500):
        processed_queries = [self.preprocessor.preprocess(q) for q in queries]
        self.ngram_vectorizer = CountVectorizer(
            analyzer='char',
            ngram_range=(2, n),
            max_features=max_features
        )
        return self.ngram_vectorizer.fit_transform(processed_queries)
    
    def create_embedding_vectors(self, queries, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        print(f"   Generating word2vec-style embeddings using SVD...")
        
        processed_queries = [self.preprocessor.preprocess(q) for q in queries]
        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=300)
        tfidf_matrix = tfidf.fit_transform(processed_queries)
        
        svd = TruncatedSVD(n_components=min(50, tfidf_matrix.shape[1] - 1), random_state=42)
        embeddings = svd.fit_transform(tfidf_matrix)
        
        self.embedding_model = "LSA (SVD-based)"
        
        return embeddings
        
        print(f"Loading embedding model: {model_name} (this may take a minute)...")
        try:
            self.embedding_model = SentenceTransformer(model_name, timeout=30)
            
            processed_queries = [self.preprocessor.preprocess(q) for q in queries]
            embeddings = self.embedding_model.encode(processed_queries, show_progress_bar=True, convert_to_numpy=True)
            
            n_components = min(100, embeddings.shape[1])
            self.pca = PCA(n_components=n_components, random_state=42)
            reduced_embeddings = self.pca.fit_transform(embeddings)
            
            return reduced_embeddings
        except Exception as e:
            print(f"   Warning: Could not load embedding model: {e}")
            print("   Using TF-IDF as fallback...")
            return self.create_tfidf_vectors(queries)

class QueryClustering:
    def __init__(self, vectors, queries):
        self.vectors = vectors
        self.queries = queries
        self.n_samples = vectors.shape[0]
        
    def kmeans_clustering(self, n_clusters=10, n_init=10, max_iter=300):
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=42
        )
        labels = kmeans.fit_predict(self.vectors)
        return labels, kmeans
    
    def hierarchical_clustering(self, n_clusters=10, linkage_method='ward'):
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method
        )
        labels = hierarchical.fit_predict(self.vectors.toarray() if hasattr(self.vectors, 'toarray') else self.vectors)
        return labels, hierarchical
    
    def dbscan_clustering(self, eps=0.5, min_samples=5):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.vectors.toarray() if hasattr(self.vectors, 'toarray') else self.vectors)
        return labels, dbscan
    
    def find_optimal_kmeans(self, k_range=range(5, 20)):
        silhouette_scores = []
        inertias = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(self.vectors)
            
            if len(set(labels)) > 1:
                sil_score = silhouette_score(self.vectors, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
            inertias.append(kmeans.inertia_)
        
        optimal_k = list(k_range)[np.argmax(silhouette_scores)]
        return optimal_k, silhouette_scores, inertias, list(k_range)

class ClusterAnalyzer:
    def __init__(self, queries, labels, vectorizer, intent_labels=None):
        self.queries = queries
        self.labels = labels
        self.vectorizer = vectorizer
        self.intent_labels = intent_labels
        self.n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
    def get_cluster_top_terms(self, n_terms=10):
        if hasattr(self.vectorizer, 'get_feature_names_out'):
            feature_names = self.vectorizer.get_feature_names_out()
        elif hasattr(self.vectorizer, 'vocabulary_'):
            feature_names = list(self.vectorizer.vocabulary_.keys())
        else:
            return {}
        
        cluster_terms = {}
        unique_labels = set(self.labels)
        
        for label in unique_labels:
            if label == -1:
                continue
            cluster_indices = np.where(self.labels == label)[0]
            cluster_queries = [self.queries[i] for i in cluster_indices]
            
            tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50)
            try:
                tfidf_matrix = tfidf.fit_transform(cluster_queries)
                scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
                top_indices = scores.argsort()[-n_terms:][::-1]
                top_terms = [tfidf.get_feature_names_out()[i] for i in top_indices]
            except:
                top_terms = []
            
            cluster_terms[label] = top_terms
        
        return cluster_terms
    
    def get_cluster_queries(self):
        cluster_queries = {}
        unique_labels = set(self.labels)
        
        for label in unique_labels:
            if label == -1:
                continue
            cluster_indices = np.where(self.labels == label)[0]
            cluster_queries[label] = [self.queries[i] for i in cluster_indices]
        
        return cluster_queries
    
    def analyze_intent_patterns(self):
        if self.intent_labels is None:
            return {}
        
        intent_counter = Counter(self.intent_labels)
        
        cluster_intents = {}
        unique_labels = set(self.labels)
        
        for label in unique_labels:
            if label == -1:
                continue
            cluster_indices = np.where(self.labels == label)[0]
            cluster_intents = Counter([self.intent_labels[i] for i in cluster_indices])
            cluster_intents[label] = dict(cluster_intents)
        
        return cluster_intents
    
    def label_clusters(self):
        cluster_terms = self.get_cluster_top_terms()
        cluster_queries = self.get_cluster_queries()
        
        cluster_labels = {}
        for label in cluster_terms.keys():
            top_terms = cluster_terms[label][:3]
            cluster_labels[label] = ' / '.join(top_terms) if top_terms else f'Cluster {label}'
        
        return cluster_labels

class Evaluator:
    def __init__(self, vectors, queries, labels, true_labels=None):
        self.vectors = vectors
        self.queries = queries
        self.labels = labels
        self.true_labels = true_labels
        
    def compute_metrics(self):
        unique_labels = set(self.labels)
        
        if len(unique_labels) <= 1 or len(unique_labels) >= len(self.queries):
            return {
                'silhouette': 0.0,
                'calinski_harabasz': 0.0,
                'davies_bouldin': float('inf')
            }
        
        try:
            vectors_dense = self.vectors.toarray() if hasattr(self.vectors, 'toarray') else self.vectors
            silhouette = silhouette_score(vectors_dense, self.labels)
            calinski = calinski_harabasz_score(vectors_dense, self.labels)
            davies = davies_bouldin_score(vectors_dense, self.labels)
            
            return {
                'silhouette': silhouette,
                'calinski_harabasz': calinski,
                'davies_bouldin': davies
            }
        except Exception as e:
            print(f"Error computing metrics: {e}")
            return {
                'silhouette': 0.0,
                'calinski_harabasz': 0.0,
                'davies_bouldin': float('inf')
            }

def plot_cluster_distribution(labels, title="Cluster Distribution"):
    unique, counts = np.unique(labels, return_counts=True)
    unique = [u for u in unique if u != -1]
    counts = [counts[i] for i, u in enumerate(unique) if u != -1]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(unique)), counts, color='steelblue')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Queries')
    plt.title(title)
    plt.xticks(range(len(unique)), [f'C{i}' for i in unique])
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{title.lower().replace(" ", "_")}.png', dpi=150)
    plt.close()

def plot_silhouette_scores(k_values, scores, title="Silhouette Scores for K-Means"):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, scores, marker='o', linewidth=2, markersize=8, color='steelblue')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    optimal_k = k_values[np.argmax(scores)]
    plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K = {optimal_k}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/silhouette_scores.png', dpi=150)
    plt.close()

def plot_pca_clusters(vectors, labels, title="PCA Visualization of Clusters"):
    if vectors.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        vectors_2d = pca.fit_transform(vectors)
    else:
        vectors_2d = vectors
    
    plt.figure(figsize=(14, 10))
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            plt.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1], 
                       c='gray', marker='x', alpha=0.5, label='Noise', s=50)
        else:
            plt.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1], 
                       c=[colors[idx]], label=f'Cluster {label}', s=80, alpha=0.7)
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/pca_clusters.png', dpi=150)
    plt.close()

def plot_dendrogram(vectors, title="Hierarchical Clustering Dendrogram"):
    plt.figure(figsize=(16, 8))
    Z = linkage(vectors.toarray() if hasattr(vectors, 'toarray') else vectors, method='ward')
    dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90, 
               leaf_font_size=10, show_contracted=True)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/dendrogram.png', dpi=150)
    plt.close()

def plot_heatmap(vectors, labels, title="Cluster Quality Heatmap"):
    metrics = []
    cluster_ids = []
    
    for label in sorted(set(labels)):
        if label == -1:
            continue
        mask = labels == label
        cluster_vectors = vectors[mask]
        
        if cluster_vectors.shape[0] > 1:
            try:
                sil = silhouette_score(cluster_vectors, [label] * cluster_vectors.shape[0])
                metrics.append({'Cluster': f'C{label}', 'Silhouette': sil})
            except:
                pass
    
    if metrics:
        df = pd.DataFrame(metrics)
        plt.figure(figsize=(12, 4))
        plt.bar(df['Cluster'], df['Silhouette'], color='steelblue')
        plt.xlabel('Cluster')
        plt.ylabel('Silhouette Score')
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/cluster_quality.png', dpi=150)
        plt.close()

def generate_report(queries, vectors, representation_name, clustering_results, output_file="report.txt"):
    with open(output_file, 'w') as f:
        f.write(f"=" * 60 + "\n")
        f.write(f"QUERY CLUSTERING REPORT\n")
        f.write(f"Intent Discovery in Web Search\n")
        f.write(f"=" * 60 + "\n\n")
        
        f.write(f"Dataset: {len(queries)} queries\n")
        f.write(f"Representation: {representation_name}\n")
        f.write(f"Vector dimensions: {vectors.shape}\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("CLUSTERING RESULTS\n")
        f.write("-" * 60 + "\n\n")
        
        for algo_name, (labels, metrics) in clustering_results.items():
            f.write(f"\n{algo_name}:\n")
            f.write(f"  Number of clusters: {len(set(labels)) - (1 if -1 in labels else 0)}\n")
            f.write(f"  Silhouette Score: {metrics.get('silhouette', 'N/A'):.4f}\n")
            f.write(f"  Calinski-Harabasz: {metrics.get('calinski_harabasz', 'N/A'):.4f}\n")
            f.write(f"  Davies-Bouldin: {metrics.get('davies_bouldin', 'N/A'):.4f}\n")
            
            cluster_queries = {}
            for label in set(labels):
                if label == -1:
                    continue
                cluster_queries[label] = [queries[i] for i in np.where(labels == label)[0]]
            
            f.write(f"\n  Cluster samples:\n")
            for label in sorted(cluster_queries.keys())[:5]:
                f.write(f"    Cluster {label}: {cluster_queries[label][:3]}\n")
            
            f.write("\n")
        
        f.write("-" * 60 + "\n")
        f.write("IMPROVEMENT ANALYSIS\n")
        f.write("-" * 60 + "\n")
        f.write("""
Query clustering can improve search systems by:
1. Intent Grouping: Similar queries are grouped, enabling better search results
2. Query Suggestion: Users typing similar queries can be offered related suggestions
3. Result Personalization: Clusters help personalize results based on user intent
4. Query Expansion: Clusters enable automatic query expansion with related terms
5. Semantic Understanding: Better understanding of user intent beyond keywords
""")

def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("QUERY CLUSTERING FOR INTENT DISCOVERY")
    print("=" * 60)
    
    print("\n[1/6] Loading dataset...")
    dataset_output = load_public_dataset()
    
    if len(dataset_output) == 3:
        queries, true_intents, countries = dataset_output
    else:
        queries, true_intents = dataset_output
        countries = ['Unknown'] * len(queries)
    
    print(f"   Loaded {len(queries)} queries")
    print(f"   Unique intents: {len(set(true_intents))}")
    print(f"   Unique countries: {len(set(countries))}")
    
    lang_counts = {'english': 0, 'german': 0, 'french': 0, 'spanish': 0}
    non_english_count = sum(1 for q in queries if any(ord(c) > 127 for c in q.lower()))
    non_english_pct = 100 * non_english_count / len(queries)

    print(f"   Multilingual queries: {non_english_count} ({non_english_pct:.1f}%)")
    
    non_english = non_english_pct
    
    print("\n[2/6] Preprocessing queries...")
    preprocessor = QueryPreprocessor()
    processed_queries = [preprocessor.preprocess(q) for q in queries]
    print(f"   Sample processed: '{queries[0]}' -> '{processed_queries[0]}'")
    
    print("\n[3/6] Creating vector representations...")
    vectorizer = QueryVectorizer(preprocessor)
    
    print("   Creating TF-IDF vectors...")
    tfidf_vectors = vectorizer.create_tfidf_vectors(queries, ngram_range=(1, 2), max_features=500)
    print(f"   TF-IDF shape: {tfidf_vectors.shape}")
    
    print("   Creating character n-gram vectors...")
    ngram_vectors = vectorizer.create_ngram_vectors(queries, n=4, max_features=500)
    print(f"   N-gram shape: {ngram_vectors.shape}")
    
    print("   Creating embedding vectors (multilingual)...")
    embedding_vectors = vectorizer.create_embedding_vectors(queries)
    print(f"   Embedding shape: {embedding_vectors.shape}")
    
    representations = {
        'TF-IDF': tfidf_vectors,
        'N-gram': ngram_vectors,
        'Embeddings': embedding_vectors
    }
    
    results = {}
    
    for rep_name, vectors in representations.items():
        print(f"\n[4/6] Clustering with {rep_name}...")
        
        clustering = QueryClustering(vectors, queries)
        clustering_results = {}
        
        print(f"   Running K-Means...")
        kmeans_labels, kmeans_model = clustering.kmeans_clustering(n_clusters=12)
        kmeans_eval = Evaluator(vectors, queries, kmeans_labels)
        kmeans_metrics = kmeans_eval.compute_metrics()
        clustering_results['K-Means'] = (kmeans_labels, kmeans_metrics)
        print(f"   K-Means silhouette: {kmeans_metrics.get('silhouette', 'N/A'):.4f}")
        
        print(f"   Running Hierarchical Clustering...")
        hier_labels, hier_model = clustering.hierarchical_clustering(n_clusters=12, linkage_method='ward')
        hier_eval = Evaluator(vectors, queries, hier_labels)
        hier_metrics = hier_eval.compute_metrics()
        clustering_results['Hierarchical'] = (hier_labels, hier_metrics)
        print(f"   Hierarchical silhouette: {hier_metrics.get('silhouette', 'N/A'):.4f}")
        
        print(f"   Running DBSCAN...")
        dbscan_labels, dbscan_model = clustering.dbscan_clustering(eps=0.5, min_samples=3)
        dbscan_eval = Evaluator(vectors, queries, dbscan_labels)
        dbscan_metrics = dbscan_eval.compute_metrics()
        clustering_results['DBSCAN'] = (dbscan_labels, dbscan_metrics)
        n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        print(f"   DBSCAN clusters: {n_dbscan_clusters}, silhouette: {dbscan_metrics.get('silhouette', 'N/A'):.4f}")
        
        results[rep_name] = clustering_results
        
        if rep_name == 'TF-IDF':
            print("\n   Creating visualizations...")
            plot_cluster_distribution(kmeans_labels, "K-Means Cluster Distribution (TF-IDF)")
            plot_pca_clusters(vectors, kmeans_labels, "PCA Visualization - K-Means (TF-IDF)")
            plot_dendrogram(vectors, "Hierarchical Clustering Dendrogram")
            plot_heatmap(vectors, kmeans_labels, "Cluster Quality Heatmap")
            
            print("   Finding optimal K...")
            optimal_k, sil_scores, inertias, k_values = clustering.find_optimal_kmeans(range(5, 20))
            plot_silhouette_scores(k_values, sil_scores)
            print(f"   Optimal K: {optimal_k}")
            
            print("\n   Cluster Analysis (TF-IDF + K-Means):")
            analyzer = ClusterAnalyzer(queries, kmeans_labels, vectorizer.tfidf_vectorizer, true_intents)
            cluster_labels = analyzer.label_clusters()
            print("\n   Cluster Labels (top terms):")
            for label, name in sorted(cluster_labels.items()):
                print(f"     Cluster {label}: {name}")
            
            cluster_queries = analyzer.get_cluster_queries()
            print("\n   Sample queries per cluster:")
            for label in sorted(cluster_queries.keys())[:5]:
                print(f"     Cluster {label}: {cluster_queries[label][:3]}")
    
    print("\n[5/6] Generating summary report...")
    generate_report(queries, tfidf_vectors, "TF-IDF", results['TF-IDF'], f"{OUTPUT_DIR}/report.txt")
    
    print("\n[6/6] Creating comparison visualizations...")
    methods = ['K-Means', 'Hierarchical', 'DBSCAN']
    silhouettes = []
    for method in methods:
        labels, metrics = results['TF-IDF'][method]
        silhouettes.append(metrics.get('silhouette', 0) or 0)
    
    plt.figure(figsize=(10, 6))
    plt.bar(methods, silhouettes, color=['steelblue', 'coral', 'seagreen'])
    plt.ylabel('Silhouette Score')
    plt.title('Clustering Algorithm Comparison (TF-IDF)')
    plt.ylim(0, 1)
    for i, v in enumerate(silhouettes):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/algorithm_comparison.png', dpi=150)
    plt.close()
    
    print("\n" + "=" * 60)
    print("COMPLETED!")
    print("=" * 60)
    print(f"\nOutput files saved to: {OUTPUT_DIR}/")
    print("  - report.txt")
    print("  - pca_clusters.png")
    print("  - dendrogram.png")
    print("  - silhouette_scores.png")
    print("  - cluster_quality.png")
    print("  - algorithm_comparison.png")

if __name__ == "__main__":
    main()