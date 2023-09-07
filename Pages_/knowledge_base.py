import streamlit as st
from nucliadb_sdk import get_or_create
from sentence_transformers import SentenceTransformer

from config import *


class FileUploader:

    def __init__(self):
        self.knowledge_base = st.sidebar.selectbox("Knowledge Base",
                                                   ["Group 1 Documents", "Group 2 Documents"])
        self.knowledge_base = "knowledge_base1" if self.knowledge_base == "Group 1 Documents" else self.knowledge_base
        self.knowledge_base = "knowledge_base2" if self.knowledge_base == "Group 2 Documents" else self.knowledge_base

    def search_knowledge_base(self):
        """
        Allow the user to insert a query, select a confidence score filter, and return the most similar text from the knowledge base.
        """
        st.subheader("Search Knowledge Base")
        query = st.text_input("Enter your query:")

        # Confidence score filter
        confidence_score = st.slider("Select a confidence score filter:", 0.0, 1.0, 0.85, 0.05)
        topk = st.radio("Select the number of top-k results:", [1, 2, 3, 4])

        if st.button("Search"):
            if query:  # Only process if there's a query
                result = self.find_most_similar(query, confidence_score, topk)
                texts = result['text']
                scores = result['score']
                sources = result['source']

                st.markdown("### Most Similar Texts")

                for text_content, score, source in zip(texts, scores, sources):
                    st.markdown(f"""
                        <div style='margin: 10px 0; padding: 10px; border: 1px solid #aaa; border-radius: 5px;'>
                            <strong>Text:</strong> {text_content}<br>
                            <strong>Score:</strong> {score}<br>
                            <strong>Source:</strong> {source}
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Please enter a query to search.")

    def find_most_similar(self, input, confidence_score, topk):
        model = SentenceTransformer(embedding_model)

        query_vectors = model.encode([input])
        my_kb = get_or_create(self.knowledge_base)
        results = my_kb.search(
            vector=query_vectors[0],
            vectorset="bge",
            min_score=confidence_score,
            page_size=topk)
        d = {"text": [result.text for result in results],
             "score": [result.score for result in results],
             "source": [result.labels[0] for result in results]}

        return d

    def search_run(self):
        self.search_knowledge_base()
