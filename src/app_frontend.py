import requests
import streamlit as st

#with st.spinner("Loading local LLM.."):
#    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", quantization_config=quantization_config)

# Title for the app
st.title('Local RAG for Podcasts')

non_rag_query = st.text_input("Query")

if st.button("Inference"):
    try:
        payload = {"query": non_rag_query}
        response = requests.post("http://localhost:5000/get_relevant_resources", json=payload)  # Replace with your actual backend URL
        context_items = response.json()["context_items"]

        payload = {"query": non_rag_query, "context_items": context_items}
        response = requests.post("http://localhost:5000/prompt_augmentation", json=payload)  # Replace with your actual backend URL
        prompt_augmented = response.json()["prompt_augmented"]

        if response.status_code == 200:
            # Display the response from the backend
            st.write("AUGMENTED PROMPT:", response.json())
        else:
            st.write(f"Failed to fetch data. Status code: {response.status_code}")

        payload = {"augmented_query": prompt_augmented, "regular_query": non_rag_query}
        response = requests.post("http://localhost:5000/inference", json=payload)  # Replace with your actual backend URL
        if response.status_code == 200:
            # Display the response from the backend
            st.write("Response from backend:", response.json())
        else:
            st.write(f"Failed to fetch data. Status code: {response.status_code}")

    except Exception as e:
        st.write(f"An error occurred: {e}")

#if st.button("Show embeddings"):
#    try:
#        payload = {"query": non_rag_query}
#        response = requests.post("http://localhost:5000/get_relevant_resources", json=payload)  # Replace with your actual backend URL
#        context_items = response.json()["context_items"]
#        if response.status_code == 200:
#            # Display the response from the backend
#            st.write("Response from backend:", response.json())
#        else:
#            st.write(f"Failed to fetch data. Status code: {response.status_code}")
#
#    except Exception as e:
#        st.write(f"An error occurred: {e}")
#    #st.text(get_embeddings(embedding_model, non_rag_query).shape)
#
#if st.button("Prompt augmentation"):
#    try:
#        payload = {"query": non_rag_query}
#        response = requests.post("http://localhost:5000/get_relevant_resources", json=payload)  # Replace with your actual backend URL
#        context_items = response.json()["context_items"]
#
#        payload = {"query": non_rag_query, "context_items": context_items}
#
#        response = requests.post("http://localhost:5000/prompt_augmentation", json=payload)  # Replace with your actual backend URL
#        if response.status_code == 200:
#            # Display the response from the backend
#            st.write("Response from backend:", response.json())
#        else:
#            st.write(f"Failed to fetch data. Status code: {response.status_code}")
#
#    except Exception as e:
#        st.write(f"An error occurred: {e}")


