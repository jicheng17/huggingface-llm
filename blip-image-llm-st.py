from transformers import pipeline
import sys
import os
import openai
from dotenv import load_dotenv
import streamlit as st

# load api key
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


def image2text(file_name):
    image_to_text = pipeline(
        "image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(file_name)[0]["generated_text"]

    print(text)
    return text


def askGPT35(query):
    prompt = f"""write me a short story about the below text
            limit the story to 100 words
            text is here ```{query}```
        """
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )
    story = response.choices[0].message["content"]
    print(story)
    return story


# create streamlit app
st.title('Image to Story')
uploaded_file = st.file_uploader(
    "Upload your image here...", type=['png', 'jpeg', 'jpg'])

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)

    st.image(uploaded_file, caption="uploaded_image.")
    print(uploaded_file.name)
    query = image2text(uploaded_file.name)
    story = askGPT35(query)
    st.write(story)
