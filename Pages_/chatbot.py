import os
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import openai
from streamlit_chat import message

import config
from utils import *


class YourDataChat:
    @staticmethod
    def how_to_gpt():
        model_used, is_api_key_valid, knowledge_base = YourDataChat.sidebar_elements()
        openai.api_key = os.environ['OPEN_AI_KEY']
        st.session_state.setdefault('cumulative_tokens', 0)
        # llm = ChatOpenAI(model_name=model_used, openai_api_key=os.environ['OPEN_AI_KEY'])
        llm = ChatOpenAI(model_name="llama2", openai_api_key="apiKey", openai_api_base="http://localhost:11434/v1")

        st.session_state.setdefault('buffer_memory', ConversationBufferWindowMemory(k=3, return_messages=True))
        system_msg_template = SystemMessagePromptTemplate.from_template(
            template=template_prompt)

        human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
        prompt_template = ChatPromptTemplate.from_messages(
            [system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

        conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm,
                                         verbose=True)

        response_container = st.container()

        query = YourDataChat.chat_input(is_api_key_valid)
        response, context = YourDataChat.generate_response(conversation, query, model_used, knowledge_base)
        YourDataChat.display_response(response_container, response)
        YourDataChat.sidebar_references(context)
        st.sidebar.button('Clear Chat History', on_click=YourDataChat.clear_text)

    @staticmethod
    def sidebar_elements():
        with st.sidebar:
            st.header("Settings")
            is_api_key_valid = True
            # add_OpenAI_api = st.text_input('Enter the OpenAI API token', type='password')
            # is_api_key_valid = add_OpenAI_api.startswith('sk') and len(add_OpenAI_api) == 51
            # if not is_api_key_valid:
            #     st.warning('Please enter your credentials', icon='⚠️')
            #
            # else:
            #     st.success('Proceed to entering your prompt message!', icon='👉')
            model_used = st.selectbox("Choose a Model", ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k'], key='select_model')
        knowledge_base = st.sidebar.selectbox("Knowledge Base",
                                              ["Group 1 Documents", "Group 2 Documents"])
        knowledge_base = "knowledge_base1" if knowledge_base == "Group 1 Documents" else knowledge_base
        knowledge_base = "knowledge_base2" if knowledge_base == "Group 2 Documents" else knowledge_base
        return model_used, is_api_key_valid, knowledge_base

    @staticmethod
    def chat_input(is_api_key_valid):
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "system", "content": config.template_prompt},
                                         {"role": "assistant", "content": "How may I assist you today?"}]

        if query := st.chat_input(disabled=not is_api_key_valid):
            st.session_state.messages.append({"role": "user", "content": query})
        return query

    @staticmethod
    def generate_response(conversation, query, model_used, knowledge_base):
        response, context = None, None
        if query:
            with st.spinner("typing.."):
                conversation_string = get_conversation_string()

                use_16k = "16k" in model_used
                context = find_match(query, use_16k, knowledge_base)

                # Dictionary to track document numbers for each source
                document_numbers = {}
                formatted_texts = []

                # Loop through the sources and texts to create the formatted string
                for source, text in zip(context['source'], context['returned_text']):
                    # If the source is not in the dictionary, assign a new number
                    if source not in document_numbers:
                        document_numbers[source] = len(document_numbers) + 1

                    # Format the text with the source and document number
                    formatted_texts.append(f'{document_numbers[source]}: {text}')

                # Join the formatted texts with newline characters
                formatted_texts_str = "\n".join(formatted_texts)

                response = conversation.predict(
                    input=f"Context:\n {formatted_texts_str} \n\n Query:\n{query}")

                # Calculate tokens and estimate price
                tokens_used = conversation_string + query + response
                tokens_used, price = num_tokens_from_string(tokens_used, model_used)

                st.session_state['cumulative_tokens'] += tokens_used
                price_estimate = (st.session_state['cumulative_tokens'] / 1000) * price

                # Display tokens and price estimate in the Streamlit app
                st.sidebar.subheader("Estimated Tokens and Price")
                st.sidebar.caption(f"🔖 Cumulative tokens used: {st.session_state['cumulative_tokens']}")
                st.sidebar.caption(f"💰 Estimated total price: ${price_estimate:.5f}")

                st.session_state.messages.append({"role": "assistant", "content": response})
        return response, context

    @staticmethod
    def display_response(response_container, response):
        with response_container:
            for i, message_data in enumerate(st.session_state.messages):
                if message_data["role"] == "assistant":
                    message(message_data["content"], key=str(i), logo=os.getenv("ROBOT_LOGO"))
                elif message_data["role"] == "user":
                    message(message_data["content"], is_user=True, key=str(i) + '_user', logo=os.getenv("USER_LOGO"))

    @staticmethod
    def sidebar_references(context):
        # Embedding the CSS
        st.sidebar.markdown("""
        <style>
            .reference-div {
                background-color: #f7f7f7; /* Light grey background */
                border: 1px solid #e1e1e1; /* Grey border */
                border-radius: 5px; /* Rounded corners */
                padding: 10px; /* Padding around the text */
                margin: 10px 0; /* Margin between each reference */
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
            }
            .pdf-text {
                font-size: 14px; /* Font size for the text */
                line-height: 1.4; /* Line spacing */
                color: #333; /* Text color */
            }
        </style>
        """, unsafe_allow_html=True)

        st.sidebar.subheader("References")
        if context:
            for text, score in zip(context['returned_text'], context['score']):
                if score >= 0.4:
                    st.sidebar.markdown(
                        f"<div class='reference-div pdf-text'>{text}</div>",
                        unsafe_allow_html=True
                    )

        st.sidebar.subheader("Document Names")
        if context:
            for doc_name, score in zip(context['source'], context['score']):
                if score >= 0.85:
                    st.sidebar.markdown(
                        f"<div class='reference-div pdf-text'>{doc_name.split('/')[-1]}</div>",
                        unsafe_allow_html=True
                    )

    @staticmethod
    def clear_text():
        st.session_state["text"] = ""
        st.session_state.messages = [{"role": "system", "content": config.template_prompt},
                                     {"role": "assistant", "content": "How may I assist you today?"}]
        st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)
