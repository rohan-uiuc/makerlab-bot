import os, uuid, json
import time
from ComplexQA import ComplexQA
import streamlit as st
from index import create_index

open_ai_key = os.environ["OPENAI_API_KEY"]  #@param {type:"string"}

# Load index

index = create_index()

# Initialize agent
functions = [index.query]
descriptions = ["This is the knowledge base to query. Only use keywords while querying this database. Do not use full sentences."]

agent = ComplexQA(open_ai_key,None, None, functions, descriptions)

# User context dictionary
user_context = {}

# Streamlit app
def main():
    st.title("Makerlab Bot")

    # Get user input
    user_input = st.text_input("Enter your query:")

    # Get user ID
    user_id = st.session_state.get("user_id", None)

    # Check if user ID exists, if not, create new ID and add to user context
    if user_id is None:
        user_id = str(uuid.uuid4())
        user_context[user_id] = {}
        print("Added user id to user context")
        st.session_state["user_id"] = user_id

    # Get user context for this user
    # print(user_context[user_id])
    context = user_context.setdefault(user_id, {})

    # Get previous query from user context
    previous_query = context.get("previous_query", "")

    # Set current query to user context
    context["previous_query"] = user_input

    # Combine current and previous query
    combined_query = previous_query + " " + user_input

    window_size = 512

    if st.button("Ask"):
        # Split the combined query into chunks of size 512
        query_chunks = [combined_query[i:i+window_size] for i in range(0, len(combined_query), window_size)]

        # Initialize the response string
        response = {}

        # Query the agent for each chunk and concatenate the response strings
        for chunk in query_chunks:
            # rate limit
            chunk_response = agent.run(chunk, max_tokens=512)
            print("type" + str(type(chunk_response)))
            response.update(chunk_response)
            time.sleep(2)

        print(response)
        final_response = response['output']['response']
        text = ''
        t = st.empty()
        for element in final_response:
            time.sleep(0.05)
            text += element
            t.markdown(text)

        context["previous_query"] = user_input
        context["previous_response"] = response
        user_context[user_id] = context

if __name__ == "__main__":
    main()
