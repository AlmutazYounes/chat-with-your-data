import streamlit as st
import os
import pickle
import faiss
import sys
sys.path.append('.')
from utils import initialize_vector_store, document_texts, document_sources, vector_store

class VectorManager:
    def __init__(self):
        initialize_vector_store()
    
    def show_vector_store_info(self):
        """Display information about the current vector store"""
        st.subheader("📊 Vector Store Information")
        
        # Import the global variables from utils
        from utils import vector_store, document_texts, document_sources
        
        if vector_store is not None and len(document_texts) > 0:
            # Get vector store statistics
            total_vectors = vector_store.ntotal
            dimension = vector_store.d
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", len(set(document_sources)))
            with col2:
                st.metric("Total Chunks", total_vectors)
            with col3:
                st.metric("Vector Dimension", dimension)
            
            # Show document sources
            st.subheader("📁 Document Sources")
            unique_sources = list(set(document_sources))
            source_counts = {}
            for source in unique_sources:
                source_counts[source] = document_sources.count(source)
            
            for source, count in source_counts.items():
                st.info(f"📄 {source}: {count} chunks")
                
        else:
            st.warning("No documents in vector store. Please upload documents first.")
    
    def view_documents(self):
        """View all documents in the vector store"""
        st.subheader("📖 View Documents")
        
        # Import the global variables from utils
        from utils import document_texts, document_sources
        
        if len(document_texts) == 0:
            st.info("No documents to display.")
            return
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            selected_source = st.selectbox(
                "Filter by document:",
                ["All Documents"] + list(set(document_sources))
            )
        
        with col2:
            search_term = st.text_input("Search in content:", placeholder="Enter search term...")
        
        # Filter documents
        filtered_texts = []
        filtered_sources = []
        filtered_indices = []
        
        for i, (text, source) in enumerate(zip(document_texts, document_sources)):
            # Apply source filter
            if selected_source != "All Documents" and source != selected_source:
                continue
            
            # Apply search filter
            if search_term and search_term.lower() not in text.lower():
                continue
            
            filtered_texts.append(text)
            filtered_sources.append(source)
            filtered_indices.append(i)
        
        # Display results
        st.write(f"**Showing {len(filtered_texts)} chunks**")
        
        for i, (text, source, idx) in enumerate(zip(filtered_texts, filtered_sources, filtered_indices)):
            with st.expander(f"Chunk {idx + 1} - {source} (Click to view)"):
                st.markdown(f"""
                <div style='padding: 15px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #e9ecef;'>
                    <div style='margin-bottom: 10px;'>
                        <strong>Source:</strong> {source}<br>
                        <strong>Chunk ID:</strong> {idx + 1}<br>
                        <strong>Length:</strong> {len(text)} characters
                    </div>
                    <div style='background-color: white; padding: 10px; border-radius: 4px; border: 1px solid #dee2e6;'>
                        {text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Delete button for individual chunk
                if st.button(f"🗑️ Delete Chunk {idx + 1}", key=f"delete_{idx}"):
                    self.delete_chunk(idx)
                    st.rerun()
    
    def delete_chunk(self, chunk_index):
        """Delete a specific chunk from the vector store"""
        # Import the global variables from utils
        import utils
        
        if 0 <= chunk_index < len(utils.document_texts):
            # Remove from lists
            del utils.document_texts[chunk_index]
            del utils.document_sources[chunk_index]
            
            # Rebuild vector store
            self.rebuild_vector_store()
            st.success(f"Chunk {chunk_index + 1} deleted successfully!")
        else:
            st.error("Invalid chunk index!")
    
    def delete_document(self, document_name):
        """Delete all chunks from a specific document"""
        # Import the global variables from utils
        import utils
        
        # Find indices to remove
        indices_to_remove = [i for i, source in enumerate(utils.document_sources) if source == document_name]
        
        if indices_to_remove:
            # Remove in reverse order to maintain indices
            for idx in reversed(indices_to_remove):
                del utils.document_texts[idx]
                del utils.document_sources[idx]
            
            # Rebuild vector store
            self.rebuild_vector_store()
            st.success(f"All chunks from '{document_name}' deleted successfully!")
        else:
            st.error(f"No chunks found for document '{document_name}'!")
    
    def rebuild_vector_store(self):
        """Rebuild the vector store from current document_texts"""
        # Import the global variables from utils
        import utils
        
        if utils.document_texts:
            from sentence_transformers import SentenceTransformer
            from config import embedding_model
            
            model = SentenceTransformer(embedding_model)
            embeddings = model.encode(utils.document_texts)
            
            # Create new vector store
            dimension = model.get_sentence_embedding_dimension()
            utils.vector_store = faiss.IndexFlatIP(dimension)
            utils.vector_store.add(embeddings.astype('float32'))
            
            # Save the updated vector store
            self.save_vector_store()
        else:
            # Create empty vector store
            from sentence_transformers import SentenceTransformer
            from config import embedding_model
            
            model = SentenceTransformer(embedding_model)
            dimension = model.get_sentence_embedding_dimension()
            utils.vector_store = faiss.IndexFlatIP(dimension)
            
            # Save the empty vector store
            self.save_vector_store()
    
    def save_vector_store(self):
        """Save the current vector store"""
        # Import the global variables from utils
        import utils
        
        if utils.vector_store is not None:
            try:
                os.makedirs('vector_store', exist_ok=True)
                faiss.write_index(utils.vector_store, 'vector_store/index.faiss')
                metadata = {
                    'texts': utils.document_texts,
                    'sources': utils.document_sources
                }
                with open('vector_store/metadata.pkl', 'wb') as f:
                    pickle.dump(metadata, f)
                st.success("Vector store saved successfully")
            except Exception as e:
                st.error(f"Error saving vector store: {e}")
    
    def clear_all_data(self):
        """Clear all data from the vector store"""
        # Import the global variables from utils
        import utils
        
        if st.button("🗑️ Clear All Data", type="primary"):
            if st.checkbox("I understand this will delete ALL documents and cannot be undone"):
                utils.document_texts = []
                utils.document_sources = []
                self.rebuild_vector_store()
                st.success("All data cleared successfully!")
                st.rerun()
    
    def export_data(self):
        """Export vector store data"""
        st.subheader("📤 Export Data")
        
        # Import the global variables from utils
        from utils import document_texts, document_sources
        
        if len(document_texts) > 0:
            # Create export data
            export_data = {
                'documents': []
            }
            
            # Group by source
            source_groups = {}
            for i, (text, source) in enumerate(zip(document_texts, document_sources)):
                if source not in source_groups:
                    source_groups[source] = []
                source_groups[source].append({
                    'chunk_id': i + 1,
                    'text': text,
                    'length': len(text)
                })
            
            for source, chunks in source_groups.items():
                export_data['documents'].append({
                    'source': source,
                    'chunk_count': len(chunks),
                    'chunks': chunks
                })
            
            # Display export info
            st.json(export_data)
            
            # Download button
            import json
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_str,
                file_name="vector_store_export.json",
                mime="application/json"
            )
        else:
            st.info("No data to export.")
    
    def run(self):
        """Main method to run the vector manager interface"""
        st.title("🗄️ Vector Store Manager")
        
        # Sidebar navigation
        st.sidebar.title("Management Options")
        option = st.sidebar.selectbox(
            "Choose an option:",
            ["📊 Overview", "📖 View Documents", "🗑️ Delete Documents", "📤 Export Data", "🗑️ Clear All"]
        )
        
        if option == "📊 Overview":
            self.show_vector_store_info()
            
        elif option == "📖 View Documents":
            self.view_documents()
            
        elif option == "🗑️ Delete Documents":
            st.subheader("🗑️ Delete Documents")
            
            # Import the global variables from utils
            from utils import document_texts, document_sources
            
            if len(document_texts) > 0:
                unique_sources = list(set(document_sources))
                
                for source in unique_sources:
                    count = document_sources.count(source)
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"📄 **{source}** ({count} chunks)")
                    
                    with col2:
                        if st.button(f"Delete", key=f"del_{source}"):
                            self.delete_document(source)
                            st.rerun()
            else:
                st.info("No documents to delete.")
                
        elif option == "📤 Export Data":
            self.export_data()
            
        elif option == "🗑️ Clear All":
            st.subheader("🗑️ Clear All Data")
            st.warning("⚠️ This will permanently delete ALL documents and chunks from the vector store!")
            self.clear_all_data() 