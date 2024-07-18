import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet
import re
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, utility

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Connect to Milvus server
connections.connect()

# Define Milvus collection and fields
embedding_dim = 384  # Dimensions of sentence embeddings from all-MiniLM-L6-v2

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
]

schema = CollectionSchema(fields, "Document embeddings collection")
collection_name = "cuda_docs"
if utility.has_collection(collection_name):
    collection = Collection(name=collection_name)
else:
    collection = Collection(name=collection_name, schema=schema)
    collection.create_index(field_name="embedding", index_params={"index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}})

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_metadata():
    with open(f"cuda_docs_metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
    return metadata

def create_bm25_index(metadata):
    tokenized_corpus = [doc[0].split() for doc in metadata]
    return BM25Okapi(tokenized_corpus)

def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    return text.lower()

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def expand_query(query, top_k=3):
    expanded_terms = []
    for word in query.split():
        synsets = wordnet.synsets(word)
        word_expanded = []
        for synset in synsets:
            for lemma in synset.lemmas():
                if lemma.name() != word and lemma.name() not in word_expanded:
                    word_expanded.append(lemma.name())
                    if len(word_expanded) == top_k:
                        break
            if len(word_expanded) == top_k:
                break
        expanded_terms.extend(word_expanded)
    return query + ' ' + ' '.join(expanded_terms)

def pseudo_relevance_feedback(query_vector, metadata, top_k=5, alpha=0.3):
    # Perform initial search
    search_params = {"metric_type": "L2", "params": {"ef": 128}}
    collection.load()
    results = collection.search(query_vector.reshape(1, -1), "embedding", search_params, limit=top_k, output_fields=["id"])
    
    # Get the top-k documents
    top_docs = [metadata[res.id][0] for res in results[0]]

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(top_docs)
    
    # Calculate centroid of top-k documents
    centroid = tfidf_matrix.mean(axis=0)
    
    # Expand query vector
    expanded_query_vector = query_vector + alpha * model.encode(vectorizer.get_feature_names_out()[centroid.argmax()])
    
    return expanded_query_vector

def hybrid_search(bm25_index, metadata, query, k=5, alpha=0.5, use_query_expansion=True, use_prf=True):
    # Preprocess and optionally expand the query
    preprocessed_query = preprocess_text(query)
    if use_query_expansion:
        expanded_query = expand_query(preprocessed_query)
    else:
        expanded_query = preprocessed_query
    
    # Dense retrieval
    query_vector = model.encode(expanded_query)
    if use_prf:
        query_vector = pseudo_relevance_feedback(query_vector, metadata)
    
    search_params = {"metric_type": "L2", "params": {"ef": 128}}
    collection.load()
    dense_results = collection.search(query_vector.reshape(1, -1), "embedding", search_params, limit=k*2, output_fields=["id"])
    dense_indices = [res.id for res in dense_results[0]]
    dense_distances = [res.distance for res in dense_results[0]]
    
    # BM25 retrieval
    bm25_scores = bm25_index.get_scores(expanded_query.split())
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:k*2]
    
    # Combine results
    combined_scores = {}
    for i, idx in enumerate(dense_indices):
        combined_scores[idx] = alpha * (1 - dense_distances[i])  # Convert distance to similarity
    
    for i, idx in enumerate(bm25_top_indices):
        if idx in combined_scores:
            combined_scores[idx] += (1 - alpha) * (bm25_scores[idx] / max(bm25_scores))
        else:
            combined_scores[idx] = (1 - alpha) * (bm25_scores[idx] / max(bm25_scores))
    
    # Sort and get top k results
    top_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)[:k]
    
    results = []
    for idx in top_indices:
        chunk, topic, url = metadata[idx]
        results.append({
            "chunk": chunk,
            "topic": topic,
            "url": url,
            "score": combined_scores[idx]
        })
    return results

def main():
    metadata = load_metadata()
    bm25_index = create_bm25_index(metadata)
    
    while True:
        query = input("Enter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        results = hybrid_search(bm25_index, metadata, query, k=5, alpha=0.5, use_query_expansion=True, use_prf=True)
        
        print("\nSearch Results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. Chunk: {result['chunk'][:100]}...")
            print(f"   Topic: {result['topic']}")
            print(f"   URL: {result['url']}")
            print(f"   Score: {result['score']}")
            print("---")

if __name__ == "__main__":
    main()