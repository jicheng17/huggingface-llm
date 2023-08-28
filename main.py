import sys
import os
import openai
from dotenv import load_dotenv
from transformers import pipeline
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
import llmclient

# load api key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Download model from Huggingface and run image to text task
def image2text(file_name):
    model_name = "Salesforce/blip-image-captioning-base"
    image_to_text = pipeline("image-to-text", model=model_name)

    text = image_to_text(file_name)[0]["generated_text"]

    print("#### image to text output: " + text)
    return text


# create streamlit app
st.title("Image to Story")
uploaded_file = st.file_uploader(
    "Upload your image here...", type=["png", "jpeg", "jpg"]
)

# wait for file upload completion
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)

    st.image(uploaded_file, caption="uploaded_image.")
    print(uploaded_file.name)
    query = image2text(uploaded_file.name)
    story = llmclient.askGPT(query)

    st.divider()
    st.write("Story about this image: ")
    st.divider()
    st.markdown(":blue[" + story + "]")
    st.divider()
