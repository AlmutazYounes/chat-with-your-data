import streamlit as st


class home_page:
    @staticmethod
    def home():
        # Header
        col1, col2 = st.columns([1, 6])
        with col1:
            st.image("static/robot.png", width=50)  # Add your logo here

        st.header("🔥 ChatBot Assistant")
        st.write("Engage in a dynamic conversation with the model to extract information from your own data.")

        st.write("Explore the sidebar to navigate between different features. Enjoy your experience!")

        st.header("Getting Started")
        st.write("To start chatting with the ChatBot Assistant, follow these steps:")
        st.write("1. Navigate to the '📁 Knowledge-Base' page in the sidebar.")
        st.write("2. Upload your documents and hit Process Files.")
        st.write("3. You can test the semantic search in '📁\U0001F50D Search Knowledge-Base' page")
        st.write("4. Navigate to the '🔥 Your Data-ChatBot' page in the sidebar and start chatting!")
