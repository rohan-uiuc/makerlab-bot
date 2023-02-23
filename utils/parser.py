# Code for the command-line argument parser

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="t5-base", help="Name of the model to use (t5-base or openai-gpt)")
    parser.add_argument("--train", type=bool, default=False, help="Whether to train the model on conversation history")
    parser.add_argument("--deploy", type=bool, default=False, help="Whether to deploy the chatbot using Streamlit")
    return parser.parse_args()
