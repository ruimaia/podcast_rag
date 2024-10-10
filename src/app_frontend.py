import requests
import streamlit as st

st.title('Local RAG for Noesis Think IT Podcast')
st.write()

non_rag_query = st.text_input("Query")

if st.button("Inference"):
    try:
        payload = {"query": non_rag_query}
        response = requests.post("http://localhost:5000/get_relevant_resources", json=payload)
        response_data = response.json()
        context_items = response_data["context_items"]

        if response.status_code == 200:
            # Display the response from the backend
            st.write("Retrieved items:", context_items)
        else:
            st.write(f"Retrieval step failed. Status code: {response.status_code}")

        payload = {"query": non_rag_query, "context_items": context_items}
        response = requests.post("http://localhost:5000/prompt_augmentation", json=payload)
        response_data = response.json()
        prompt_augmented = response_data["prompt_augmented"]

        if response.status_code == 200:
            # Display the response from the backend
            st.write("Prompt augmented:", response_data)
        else:
            st.write(f"Failed to add context to original prompt. Status code: {response.status_code}")

        payload = {"augmented_query": prompt_augmented, "regular_query": non_rag_query}
        response = requests.post("http://localhost:5000/inference", json=payload)
        response_data = response.json()
        if response.status_code == 200:
            # Display the response from the backend
            st.write("Generation:", response_data)
        else:
            st.write(f"Generation step failed. Status code: {response.status_code}")

    except Exception as e:
        st.write(f"An error occurred: {e}")

