# Question-Answering-System

This application leverages a hybrid search method combining dense vector search and BM25-based retrieval powered by Sentence Transformers and Milvus(or Faiss) to provide accurate and relevant answers to questions about CUDA documentation. It also uses OpenAI’s GPT-3.5 for generating answers from the search results.

### Working gradio application:

https://github.com/user-attachments/assets/24e83cae-07ed-4f83-bcd7-fcc246e3d58f


# Features

### Document Processing

•	Scraping and Chunking: The system can scrape CUDA documentation from official sources automatically splitting it into manageable chunks for indexing and retrieval. This preprocessing step ensures that the search index is comprehensive and up-to-date.
•	Metadata Handling: Each chunk of documentation is stored with relevant metadata such as the topic and URL facilitating more detailed and informative search results.



### Hybrid Search

•	Dense Vector Search: Uses Milvus or FAISS to perform efficient and scalable dense vector searches. This allows for high-speed retrieval of relevant documents based on semantic similarity.
•	BM25 Retrieval: Employs the BM25 algorithm a popular probabilistic IR model to retrieve documents based on term frequency and inverse document frequency ensuring accurate keyword-based search results.

### Query Expansion

•	WordNet-Based Expansion: Utilizes WordNet to expand queries by adding synonyms and related terms improving the recall and breadth of search results.

### Pseudo Relevance Feedback

•	Enhanced Query Vectors: Refines the initial query vector by leveraging the top results from the initial search. This feedback loop helps in adjusting the query vector to better match relevant documents.

### Answer Generation

•	OpenAI GPT-3.5: Integrates OpenAI’s GPT-3.5 to generate answers based on retrieved passages. This ensures that answers are coherent, contextually relevant, and informative.

### Interactive Interface

•	Gradio Integration: Provides an easy-to-use web interface powered by Gradio. Users can input queries, view search results, and receive answers in a streamlined manner.
•	Real-Time Interaction: Offers real-time interaction, allowing users to quickly query and receive responses about CUDA documentation.

# Installation Guide

Download all the dependencies from the requirements.txt file.
```
pip unstall -r requirements.txt
```
# Running the System

### OR


•	Clone the repository in your own system and run all the .py files in the terminal or you can even directly run my .ipynd file in jupyter notebook or google colab.
•	Make sure to use your OpenAI key while running the LLM.




