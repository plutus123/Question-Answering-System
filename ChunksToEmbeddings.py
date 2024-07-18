import gc
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
import re
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = SentenceTransformer('all-MiniLM-L6-v2')
model.to(device)

def chunk_generator(file_path, batch_size=1000):
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = []
        for line in f:
            chunks.append(line.strip())
            if len(chunks) == batch_size:
                yield chunks
                chunks = []
        if chunks:
            yield chunks

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return [word for word in text.split() if word not in STOPWORDS]

def create_topic_model(chunk_gen, num_topics=10):
    dictionary = corpora.Dictionary()
    corpus = []
    for chunks in chunk_gen:
        preprocessed_chunks = [preprocess_text(chunk) for chunk in chunks]
        dictionary.add_documents(preprocessed_chunks)
        corpus.extend([dictionary.doc2bow(text) for text in preprocessed_chunks])
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)
    return lda_model, dictionary

def get_chunk_topic(chunk, lda_model, dictionary):
    bow = dictionary.doc2bow(preprocess_text(chunk))
    topics = lda_model.get_document_topics(bow)
    return max(topics, key=lambda x: x[1])[0] if topics else None

@torch.no_grad()
def process_and_save_chunks(chunks, lda_model, dictionary, output_file, batch_size=64):
    with open(output_file, 'a', encoding='utf-8') as f:
        for i in tqdm(range(0, len(chunks), batch_size), desc="Processing batches"):
            batch_chunks = chunks[i:i + batch_size]
            try:
                embeddings = model.encode(batch_chunks, convert_to_tensor=True, device=device)
                topics = [get_chunk_topic(chunk, lda_model, dictionary) for chunk in batch_chunks]
                
                for chunk, embedding, topic in zip(batch_chunks, embeddings, topics):
                    embedding_str = ' '.join(map(str, embedding.cpu().numpy().tolist()))
                    f.write(f"{chunk}\t{embedding_str}\t{topic}\n")
            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {e}")
            
            torch.cuda.empty_cache()

if __name__ == '__main__':
    chunked_text_file = 'chunked_text.txt'
    output_file = 'chunked_data_with_embeddings.txt'
    
    try:
        print("Creating topic model...")
        lda_model, dictionary = create_topic_model(chunk_generator(chunked_text_file))
        
        print("Processing chunks and saving data...")
        for chunks in chunk_generator(chunked_text_file):
            process_and_save_chunks(chunks, lda_model, dictionary, output_file, batch_size=64)
            gc.collect()
        
        print(f'Embedding conversion and topic modeling completed. Data saved to {output_file}')
    except Exception as e:
        print(f"Error: {e}")