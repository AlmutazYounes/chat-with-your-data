import re

import streamlit as st
import tiktoken
from dotenv import load_dotenv
from nucliadb_sdk import get_or_create
from pdf2image import convert_from_bytes
from pdfminer.high_level import extract_text
from pytesseract import image_to_string, pytesseract
from sentence_transformers import SentenceTransformer

from config import *

load_dotenv()
model = SentenceTransformer(embedding_model)


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


def find_match(input, turbo_16k, knowledge_base):
    confidence_threshold = 0.5
    reference_number = 4
    query_vectors = model.encode([input])
    my_kb = get_or_create(knowledge_base)
    results = my_kb.search(
        vector=query_vectors[0],
        vectorset="bge",
        min_score=confidence_threshold,
        page_size=reference_number)

    context = {"returned_text": [],
               "source": [],
               "score": []}
    total_tokens = 0
    max_tokens_allowed = 16385 if turbo_16k else 4097
    for result in results:
        if total_tokens < max_tokens_allowed:
            context["returned_text"].append(result.text)
            context["source"].append(result.labels[0])
            context["score"].append(result.score)
            total_tokens += len(result.text)
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
            rotate_angle = int(re.search('(?<=Rotate: )\d+', osd).group(0))
        except:
            return image  # If the rotation angle can't be determined, return the original image

        # If rotation is detected, correct it
        if rotate_angle:
            return image.rotate(360 - rotate_angle, expand=True)
        return image

    def extract_text_from_image_pdf(self, pdf_file):
        # Convert PDF to list of images
        pdf_bytes = pdf_file.read()
        images = convert_from_bytes(pdf_bytes)

        # Display a message that the image-based PDF is being rendered
        st.subheader("We are reading your image-based PDF...")

        # Create a progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()  # Placeholder for the progress text

        # Correct rotation and extract text from each image
        extracted_texts = []
        for index, img in enumerate(images):
            corrected_img = self.correct_rotation(img)
            extracted_texts.append(image_to_string(corrected_img))

            # Update the progress bar
            progress = (index + 1) / len(images)
            progress_bar.progress(progress)

            # Update the progress text with the percentage
            progress_text.text(f"Progress: {progress * 100:.2f}%")

        # Combine texts from all pages
        full_text = "\n".join(extracted_texts)
        return full_text

    def read_pdf(self):
        extracted_text = extract_text(self.file)

        # If the extracted text size is very small (e.g., less than 100 bytes),
        # assume it's an image-based PDF and process accordingly
        if len(extracted_text) < 100:
            self.file.seek(0)
            extracted_text = self.extract_text_from_image_pdf(self.file)

        return extracted_text
