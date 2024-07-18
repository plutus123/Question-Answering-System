from openai import OpenAI
import os
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

# Set your OpenAI API key directly here
openai_key = "OPENAI_KEY"

# Initialize OpenAI client
client = OpenAI(api_key=openai_key)

# Initialize SentenceTransformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def prepare_context(results: List[Dict], query: str) -> str:
    """Prepare the context for the LLM from the search results, ranking by relevance to the query."""
    # Encode the query and chunks
    query_embedding = sentence_model.encode(query)
    chunk_embeddings = sentence_model.encode([result['chunk'] for result in results])
    
    # Calculate cosine similarities
    similarities = np.dot(chunk_embeddings, query_embedding) / (np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding))
    
    # Sort results by similarity
    sorted_results = [result for _, result in sorted(zip(similarities, results), key=lambda x: x[0], reverse=True)]
    
    context = "Here are some relevant passages from the CUDA documentation, ordered by relevance:\n\n"
    for i, result in enumerate(sorted_results, 1):
        context += f"{i}. {result['chunk']}\n\n"
    return context

def answer_question(query: str, results: List[Dict]) -> str:
    """Use GPT to answer the question based on the retrieved and ranked results."""
    context = prepare_context(results, query)
    
    messages = [
        {"role": "system", "content": """You are a helpful assistant that answers questions about CUDA based on the provided context. 
        Follow these guidelines:
        1. Always base your answers on the information provided in the context.
        2. If the answer cannot be found in the context, clearly state that you don't have enough information to answer accurately.
        3. If the context contains conflicting information, mention this and explain the different viewpoints.
        4. Use technical terms correctly and explain them if they're complex.
        5. If appropriate, structure your answer with bullet points or numbered lists for clarity.
        6. Cite the relevant passage numbers from the context to support your answer.
        7. If the user's question is unclear, ask for clarification before attempting to answer."""},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
    ]
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.3,  # Lower temperature for more focused answers
    )
    
    return response.choices[0].message.content.strip()

def main():
    index_type = "FLAT"  # or "IVF"
    dense_index, metadata = load_index_and_metadata(index_type)  # Assuming these functions are defined elsewhere
    bm25_index = create_bm25_index(metadata)  # Assuming this function is defined elsewhere
    
    while True:
        query = input("Enter your question about CUDA (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        results = hybrid_search(dense_index, bm25_index, metadata, query, k=5, alpha=0.5, use_query_expansion=True, use_prf=True)
        
        print("\nRetrieved Passages:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['chunk'][:100]}...")
        
        answer = answer_question(query, results)
        print("\nAnswer:")
        print(answer)
        print("---")

if __name__ == "__main__":
    main()