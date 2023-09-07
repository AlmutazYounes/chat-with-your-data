embedding_model = 'BAAI/bge-base-en'
template_prompt = """
You will receive one or more documents along with a user's input. Generate an answer for the user's question and follow these instructions:
1. If no question is asked, respond in a friendly manner.
2. If a question is asked, answer it truthfully using the provided text for reference.
3. If the answer is not in the text, respond with 'I don't know.'
4. Mention reference number. It must be mentioned once at the end of the generated text from the each document.
Example: What are .....?
Answer: They are .....[¹]
References:
[¹] file name
"""
