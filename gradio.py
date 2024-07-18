import gradio as gr
from QALLM import answer
from VectorDatabase-Retrieval-Reranking import hybrid_search
def main():
    index_type = "FLAT"  # or "IVF"
    dense_index, metadata = load_index_and_metadata(index_type)  # Assuming these functions are defined elsewhere
    bm25_index = create_bm25_index(metadata)  # Assuming this function is defined elsewhere
    
    def inference(query):
        nonlocal dense_index, bm25_index, metadata
        results = hybrid_search(dense_index, bm25_index, metadata, query, k=5, alpha=0.5, use_query_expansion=True, use_prf=True)
        answer = answer_question(query, results)
        return answer

    iface = gr.Interface(
        fn=inference,
        inputs="text",
        outputs="text",
        title="CUDA Documentation Assistant",
        description="Ask a question about CUDA documentation.",
        theme="huggingface",
        examples=[["How to use CUDA with Python?"]],
    )
    iface.launch(share=True)

if __name__ == "__main__":
    main()








