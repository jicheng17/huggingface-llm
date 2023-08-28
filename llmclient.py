import sys
import os
import openai
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


# load api key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Query openAI LLM using openAI library
def askGPT(query):
    llm = OpenAI(temperature=0.7, openai_api_key=openai.api_key)
    template = """write me a short story about the below text
            limit the story to 100 words
            text is here {query}
        """

    prompt = PromptTemplate(template=template, input_variables=["query"])

    story = llm(prompt.format(query=query))
    print("story is : " + story)
    return story
