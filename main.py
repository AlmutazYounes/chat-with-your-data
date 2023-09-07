import streamlit as st

from Pages_.chatbot import YourDataChat
from Pages_.home import home_page
from Pages_.knowledge_base import FileUploader


def main():
    st.sidebar.title("Pages")
    page = st.sidebar.selectbox("", ["🏠 Home", "📁\U0001F50D Search Knowledge-Base", "🔥 Your Data-ChatBot"])

    if page == "🏠 Home":
        home_page.home()

    elif page == "📁\U0001F50D Search Knowledge-Base":
        file_uploader = FileUploader()
        file_uploader.search_run()


    elif page == "🔥 Your Data-ChatBot":
        YourDataChat.how_to_gpt()


def main2():
    st.set_page_config(layout="wide")
    YourDataChat.how_to_gpt()


if __name__ == "__main__":
    main()
    # main2()
