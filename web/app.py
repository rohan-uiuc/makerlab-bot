import streamlit as st
from transformers import TFAutoModelForCausalLM, TFT5Model, T5ForConditionalGeneration, AutoTokenizer, T5Tokenizer, OpenAIGPTTokenizer, pipeline
import json
import config.config
from web import training
from web.langchain import LangChainDB
from utils.scraper import get_contents


# Load the T5 and GPT-2 models
T5_MODEL_NAME = "t5-small"
GPT_MODEL_NAME = "openai-gpt"
st.session_state["model"] = "t5"

# with open(config.config.T5_CONFIG_FILE,) as f:
#     try:
#         t5_config = json.load(f)
#         print("File is valid JSON!")
#     except ValueError as e:
#         print("File is not valid JSON: ", e)


t5_model = TFT5Model.from_pretrained(T5_MODEL_NAME, config=config.config.T5_CONFIG_FILE, ignore_mismatched_sizes=True)
gpt_model = TFAutoModelForCausalLM.from_pretrained(GPT_MODEL_NAME)


t5_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
gpt_tokenizer = OpenAIGPTTokenizer.from_pretrained(GPT_MODEL_NAME)

# t5_model.save_pretrained(config.config.T5_MODEL_DIR)
# t5_tokenizer.save_pretrained(config.config.T5_MODEL_DIR)
#
# gpt_model.save_pretrained(config.config.GPT_MODEL_DIR)
# gpt_tokenizer.save_pretrained(config.config.GPT_MODEL_DIR)

# Define the conversation history database
DB_NAME = "conversation_history.db"
db = LangChainDB(DB_NAME)

# Define the model switcher
models = {
    "t5": {"model": t5_model, "tokenizer": t5_tokenizer},
    "gpt": {"model": gpt_model, "tokenizer": gpt_tokenizer}
}

# Define the chatbot pipeline
def get_chatbot_pipeline(model_name):
    model = models[model_name]["model"]
    tokenizer = models[model_name]["tokenizer"]
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Define the conversation history key
def get_history_key():
    return st.session_state.user_input

# Define the chatbot response generator
def generate_response(chatbot_pipeline, user_input, context=None):
    if context:
        user_input = context + " " + user_input
    response = chatbot_pipeline(user_input)[0]["generated_text"]
    return response.strip()

def set_radio_callback(widgetKey, stateKey):
    st.session_state[stateKey] = st.session_state[widgetKey]

# Define the Streamlit app
def app():
    get_contents(config.config.INDEX_URL)
    # Initialize the user input and conversation history
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = {}

    # Retrieve the conversation history for the current user input
    history_key = get_history_key()
    conversation_history = db.retrieve(history_key)

    training.train_model(t5_model, t5_tokenizer, config.config.DATA_MODEL_DIR)
    training.train_model(gpt_model, gpt_tokenizer, config.config.DATA_MODEL_DIR)

    # Define the title and header of the app
    st.title("Chatbot")
    st.header("Enter a prompt and the chatbot will generate a response.")

    # Define the user input form and submit button
    st.subheader("Prompt:")
    user_input = st.text_input(label="", value=st.session_state.user_input)
    submit_button = st.button("Submit")

    # Handle user input submission
    if submit_button:
        # Retrieve the chatbot pipeline and generate a response
        chatbot_pipeline = get_chatbot_pipeline(st.session_state.model)
        response = generate_response(chatbot_pipeline, user_input, conversation_history)

        # Store the conversation history for the current user input
        if history_key not in st.session_state.conversation_history:
            st.session_state.conversation_history[history_key] = []
        st.session_state.conversation_history[history_key].append((user_input, response))
        db.store(history_key, st.session_state.conversation_history[history_key])

        # Display the chatbot response
        st.subheader("Response:")
        st.write(response)

    # Define the model switcher form and submit button
    st.subheader("Model:")
    model = st.radio(label="model", options=["t5", "gpt"], index=["t5", "gpt"].index(st.session_state.model), key="model", on_change=set_radio_callback, args=("model", "model"))
    switch_button = st.button("Switch")

    # Handle model switch submission
    if switch_button:
        # Reset the user input and conversation history
        st.session_state.user_input = ""
        st.session_state.conversation_history = {}

        # Update the current model
        st.session_state.model = model
        st.session_state.chatbot_pipeline = get_chatbot_pipeline(model)

if __name__ == '__main__':
    app()
