import streamlit as st
import os
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import openai
from utils import find_match, get_conversation_string, num_tokens_from_string

class YourDataChat:
    def __init__(self):
        self.model_name = "gpt-4o-mini"  # Using gpt-4o-mini as gpt-4.1-nano might not be available yet
        self.api_key = os.environ.get('OPEN_AI_KEY')
        
    def get_response(self, user_input):
        """Get a response from the chatbot for a given input"""
        try:
            if not self.api_key:
                return "Please set your OpenAI API key in the settings to use the chat functionality."
            
            # Get context from knowledge base
            context = self.get_context(user_input)
            
            # Generate response using OpenAI
            response = self.generate_openai_response(user_input, context)
            
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_context(self, query):
        """Get relevant context from the knowledge base"""
        try:
            # Use the simplified find_match function
            results = find_match(query, top_k=4)
            return results
        except Exception as e:
            st.error(f"Error getting context: {e}")
            return {"returned_text": [], "source": [], "score": []}
    
    def generate_openai_response(self, user_input, context):
        """Generate response using OpenAI API"""
        try:
            # Prepare context
            context_text = ""
            if context and context.get('returned_text'):
                context_text = "\n\n".join(context['returned_text'])
            
            # Create system prompt
            system_prompt = """You are a helpful AI assistant that answers questions based on the provided context from documents. 
            
            IMPORTANT INSTRUCTIONS:
            1. If context is provided, analyze it carefully for ANY relevant information
            2. Even if the context seems limited, try to extract useful insights from it
            3. If the context contains figures, tables, or technical content, explain what they show
            4. If you find ANY relevant information, provide a detailed answer based on it
            5. Only say "I don't have information about this" if the context is completely irrelevant
            6. Always mention that you're answering based on the uploaded documents when you use them
            7. Be specific and cite information from the documents when possible
            8. If the context shows figures or tables, explain their significance
            
            Always be helpful, accurate, and try to provide value even from limited context."""
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add context if available
            if context_text:
                messages.append({
                    "role": "user", 
                    "content": f"Context from uploaded documents:\n{context_text}\n\nQuestion: {user_input}"
                })
            else:
                messages.append({
                    "role": "user", 
                    "content": f"Question: {user_input}\n\nNote: No relevant information found in uploaded documents."
                })
            
            # Call OpenAI API using new format
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def get_conversation_string(self):
        """Get conversation history as string"""
        conversation_string = ""
        if "messages" in st.session_state:
            for message_data in st.session_state.messages:
                role = "Human" if message_data["role"] == "user" else "Bot"
                conversation_string += f"{role}: {message_data['content']}\n"
        return conversation_string
