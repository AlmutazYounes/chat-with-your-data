import re
import os
import pickle
import streamlit as st
import tiktoken
from dotenv import load_dotenv
import faiss
import numpy as np
from pdf2image import convert_from_bytes
from pdfminer.high_level import extract_text
from pytesseract import image_to_string, pytesseract
from sentence_transformers import SentenceTransformer
import warnings
import logging
import nltk

# Download `punkt` tokenizer if not already present
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)

# Suppress PDF processing warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('pdfminer').setLevel(logging.ERROR)

from config import *

load_dotenv()
model = SentenceTransformer(embedding_model)

# Global variables for vector store
vector_store = None
document_texts = []
document_sources = []

def clear_vector_store():
    """Completely clear the vector store and all data from session_state"""
    keys_to_delete = ['vector_store', 'document_texts', 'document_sources', 'processed_files']
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    
    # Re-initialize to ensure the app is in a clean state
    initialize_vector_store()
    st.success("Vector store has been cleared.")


def initialize_vector_store():
    """Initialize a fresh FAISS vector store in session_state if it doesn't exist"""
    if 'vector_store' not in st.session_state:
        dimension = model.get_sentence_embedding_dimension()
        st.session_state.vector_store = faiss.IndexFlatIP(dimension)
        st.session_state.document_texts = []
        st.session_state.document_sources = []
        st.session_state.processed_files = set()


def add_documents_to_vector_store(texts, source):
    """Add documents to the vector store held in session_state"""
    if 'vector_store' not in st.session_state:
        initialize_vector_store()
    
    # Debug: Initial processing info
    print(f"🔧 Starting chunking process for '{source}' with {len(texts)} text sections")
    if 'st' in globals():
        st.info(f"🔧 Starting chunking process for '{source}' with {len(texts)} text sections")
    
    # Split texts into chunks based on token count
    chunks = []
    total_sentences = 0
    
    for i, text in enumerate(texts):
        # Improved sentence splitting using NLTK
        sentences = nltk.sent_tokenize(text)
        total_sentences += len(sentences)
        
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Add period back if it was removed
            if not sentence.endswith('.'):
                sentence += '.'
            
            # Count tokens for this sentence
            sentence_tokens, _ = num_tokens_from_string(sentence, 'gpt-3.5-turbo')
            
            # Check if adding this sentence would exceed 500 tokens
            if current_tokens + sentence_tokens <= 500:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sentence_tokens
                
                # If we've reached 200+ tokens, create a chunk (lowered from 300)
                if current_tokens >= 200:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    current_tokens = 0
            else:
                # Current chunk would exceed 500 tokens, save it if it has content
                if current_chunk.strip():  # Save any chunk with content
                    chunks.append(current_chunk)
                
                # Start new chunk with this sentence
                current_chunk = sentence
                current_tokens = sentence_tokens
        
        # Add the last chunk if it has content (lowered threshold)
        if current_chunk.strip():  # Save any remaining content
            chunks.append(current_chunk)
    
    # Debug: Chunking results
    print(f"✅ Chunking complete! Created {len(chunks)} chunks from {total_sentences} sentences")
    if 'st' in globals():
        st.success(f"✅ Chunking complete! Created {len(chunks)} chunks from {total_sentences} sentences")
        if chunks:
            avg_chunk_length = sum(len(chunk) for chunk in chunks) / len(chunks)
            st.info(f"📊 Average chunk length: {avg_chunk_length:.0f} characters")
    
    if chunks:
        try:

            
            # Generate embeddings
            embeddings = model.encode(chunks)
            
            # Add to vector store in session_state
            st.session_state.vector_store.add(embeddings.astype('float32'))
            
            # Store metadata in session_state
            st.session_state.document_texts.extend(chunks)
            st.session_state.document_sources.extend([source] * len(chunks))
            
            # Debug: Final vector store status
            print(f"🎯 Vector store updated! Total documents: {len(set(st.session_state.document_sources))}, Total chunks: {len(st.session_state.document_texts)}")
            if 'st' in globals():
                st.success(f"🎯 Vector store updated! Total documents: {len(set(st.session_state.document_sources))}, Total chunks: {len(st.session_state.document_texts)}")
            
            return True
        except Exception as e:
            print(f"❌ Error adding to vector store: {e}")
            if 'st' in globals():
                st.error(f"❌ Error adding to vector store: {e}")
            return False
    else:
        print(f"❌ No chunks created - text might be too short")
        if 'st' in globals():
            st.warning(f"❌ No chunks created - text might be too short")
        return False

def num_tokens_from_string(string: str, model_used: str):
    """Returns the number of tokens in a text string."""
    if model_used == 'gpt-3.5-turbo':
        price, encoding_name = 0.002, "cl100k_base"
    elif model_used == 'gpt-3.5-turbo-16k':
        price, encoding_name = 0.004, "cl100k_base"
    elif model_used == 'text-davinci-003':
        price, encoding_name = 0.02, "p50k_base"
    else:
        price, encoding_name = 0.02, "p50k_base"
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens, price

def find_match(input, top_k=6, knowledge_base="default"):
    """Find similar documents using FAISS vector search from session_state"""
    if 'vector_store' not in st.session_state or len(st.session_state.document_texts) == 0:
        return {"returned_text": [], "source": [], "score": []}
    
    confidence_threshold = 0.3  # Much lower threshold to get more results
    reference_number = top_k
    
    # Generate query embedding
    query_embedding = model.encode([input])
    
    # Search in vector store from session_state
    scores, indices = st.session_state.vector_store.search(query_embedding.astype('float32'), reference_number)
    
    context = {"returned_text": [], "source": [], "score": []}
    
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < len(st.session_state.document_texts) and score >= confidence_threshold:
            text = st.session_state.document_texts[idx]
            context["returned_text"].append(text)
            context["source"].append(st.session_state.document_sources[idx])
            context["score"].append(float(score))
    
    return context

def get_conversation_string():
    conversation_string = ""
    for message_data in st.session_state.messages:
        role = "Human" if message_data["role"] == "user" else "Bot"
        conversation_string += f"{role}: {message_data['content']}\n"
    return conversation_string

class Handle_pdf:
    def __init__(self, file):
        self.file = file

    @staticmethod
    def correct_rotation(image):
        # Use pytesseract to detect orientation
        try:
            osd = pytesseract.image_to_osd(image)
            rotate_angle = int(re.search(r'(?<=Rotate: )\d+', osd).group(0))
        except:
            return image  # If the rotation angle can't be determined, return the original image

        # If rotation is detected, correct it
        if rotate_angle:
            return image.rotate(360 - rotate_angle, expand=True)
        return image

    def extract_text_from_image_pdf(self, pdf_file):
        try:
            # Convert PDF to list of images
            pdf_bytes = pdf_file.read()
            images = convert_from_bytes(pdf_bytes, quiet=True)

            # Correct rotation and extract text from each image
            extracted_texts = []
            for index, img in enumerate(images):
                try:
                    corrected_img = self.correct_rotation(img)
                    text = image_to_string(corrected_img, config='--quiet')
                    if text.strip():
                        extracted_texts.append(text)
                except Exception as e:
                    # Continue processing other pages even if one fails
                    continue

            return extracted_texts
        except Exception as e:
            st.error(f"Error processing image-based PDF: {e}")
            return []

    def read_pdf(self):
        try:
            # Debug: Starting PDF reading
            print(f"📖 Reading PDF: {self.file}")
            
            # First, try to extract text directly
            pdf_text = extract_text(self.file)
            
            if pdf_text and pdf_text.strip():  # If text extraction was successful
                print(f"✅ Text extraction successful! Extracted {len(pdf_text)} characters")
                if 'st' in globals():
                    st.success(f"✅ Text extraction successful! Extracted {len(pdf_text)} characters")
                return [pdf_text]
            else:  # If no text was extracted, it might be an image-based PDF
                print(f"⚠️ No text found, attempting OCR for image-based PDF...")
                if 'st' in globals():
                    st.warning(f"⚠️ No text found, attempting OCR for image-based PDF...")
                
                # For now, skip OCR and return empty - we can fix OCR later
                print(f"❌ OCR not implemented yet - no text could be extracted")
                if 'st' in globals():
                    st.error(f"❌ No text found in PDF. This might be an image-based PDF requiring OCR.")
                return []
                
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            if 'st' in globals():
                st.error(f"❌ Error reading PDF: {e}")
            return []
