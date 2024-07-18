import spacy
from sentence_transformers import SentenceTransformer
import numpy as np

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 1000000  # Reduced max length for faster processing
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_data(file_path, max_pages=5):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    # Split the data by the separator used in the scraped data file
    pages = data.split("\n" + "-"*80 + "\n")
    parsed_data = []
    
    for page in pages[:max_pages]:  # Limit to max_pages
        if page.strip():
            parts = page.split("\n", 1)
            url = parts[0].replace("URL: ", "")
            text = parts[1] if len(parts) > 1 else ""
            parsed_data.append({'url': url, 'text': text})
    
    return parsed_data

def chunk_data(scraped_data, max_sentences=50):
    chunked_data = []
    
    for page_data in scraped_data:
        chunks = []
        text = page_data.get('text', '')
        
        # Process text using spaCy for sentence segmentation
        doc = nlp(text[:100000])  # Limit text length for faster processing
        
        # Chunk sentences based on similarity
        current_chunk = []
        for sent in list(doc.sents)[:max_sentences]:  # Limit number of sentences
            if current_chunk:
                # Check semantic similarity between current chunk and new sentence
                chunk_embedding = model.encode(" ".join([str(s) for s in current_chunk]))
                sent_embedding = model.encode(sent.text)
                similarity = np.dot(chunk_embedding, sent_embedding) / (np.linalg.norm(chunk_embedding) * np.linalg.norm(sent_embedding))
                
                if similarity < 0.7:  # Threshold for semantic similarity
                    chunks.append(" ".join([str(s) for s in current_chunk]))
                    current_chunk = []
            
            current_chunk.append(sent)
        
        if current_chunk:
            chunks.append(" ".join([str(s) for s in current_chunk]))
        
        page_data['chunks'] = chunks
        chunked_data.append(page_data)
    
    return chunked_data

def save_chunked_data(chunked_data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for page_data in chunked_data:
            for chunk in page_data['chunks']:
                f.write(chunk + '\n')

if __name__ == '__main__':
    input_file = 'scraped_data.txt'
    chunked_text_file = 'chunked_text.txt'
    
    # Load scraped data (limited to 5 pages)
    scraped_data = load_data(input_file, max_pages=5)
    
    # Chunk the data based on semantic similarity or topics (limited to 50 sentences per page)
    chunked_data = chunk_data(scraped_data, max_sentences=50)
    
    # Save chunked data to text file
    save_chunked_data(chunked_data, chunked_text_file)
    
    print(f'Chunking completed. Data saved to {chunked_text_file}')