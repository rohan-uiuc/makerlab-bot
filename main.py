import os


import streamlit as st
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.llms import HuggingFaceHub
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma

from langchain.llms import OpenAI
from langchain.chains import VectorDBQA

from utils import get_search_index
from utils import generate_answer
from langchain.chains import VectorDBQA, VectorDBQAWithSourcesChain

open_ai_pkl = "open_ai.pkl"
open_ai_index = "open_ai.index"
ul2_pkl = "ul2.pkl"
ul2_index = "ul2.index"





gpt_3_5 = OpenAI(model_name='gpt-3.5-turbo',temperature=0)
flan_ul2 = HuggingFaceHub(repo_id="google/flan-ul2", model_kwargs={"temperature":0.1, "max_tokens":10000, "max_length": 10000})

open_ai_embeddings = OpenAIEmbeddings()
hf_embeddings = HuggingFaceHubEmbeddings()


def run():

    # Get user input
    st.title("MakerlabX3DPrinting QA")
    question = st.text_input("Enter your question:")

    gpt_3_5_index = get_search_index(open_ai_pkl, open_ai_index, open_ai_embeddings)
    hf_search_index = get_search_index(ul2_pkl, ul2_index, hf_embeddings)

    gpt_3_5_chain = load_qa_with_sources_chain(gpt_3_5, chain_type="stuff")
    flan_ul2_chain = load_qa_with_sources_chain(flan_ul2, chain_type="stuff")

    # print("Print index size " + str(len(hf_search_index.index)))

    if st.button("UL2 Answer"):
        qa = VectorDBQAWithSourcesChain(
            combine_documents_chain=flan_ul2_chain,
            vectorstore=hf_search_index,
            reduce_k_below_max_tokens=False,
        )
        answer = qa(question)
        show_answer(answer)
    if st.button("GPT Answer"):
        answer = generate_answer(gpt_3_5_chain, gpt_3_5_index, question)
        show_answer(answer)


def show_answer(answer):
    # Show answer
    st.write(f"Answer: {answer}")


if __name__ == '__main__':
    run()