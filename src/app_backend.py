# app_backend.py
from flask import Flask, request, jsonify
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import generate_embeddings

app = Flask(__name__)

# Load the model once at app startup
embedding_model = None
llm = None

# Load the models at startup
#quantization_config = BitsAndBytesConfig(load_in_4bit=True)
#tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
#llm = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", quantization_config=quantization_config)
#
embedding_model = generate_embeddings.load_model(generate_embeddings.MODEL_ID, device="cuda")
print("Finished loading the models..")

@app.route('/foo', methods=['GET'])
def foo():
    # Use the JSON data in your logic
    return jsonify({
        "message": f"Hello world",
    })

@app.route('/get_embeddings', methods=['POST'])
def get_embeddings():
    # Access JSON data using request.json
    data = request.json
    query = data.get('query')
    embeddings = generate_embeddings.get_embeddings(embedding_model, query)

    # Use the JSON data in your logic
    return jsonify({
        "query": f"Query: {query}",
        "embeddings": f"embeddings: {embeddings}"
    })

#@app.route('/predict', methods=['POST'])
#def predict():
#    # Access JSON data using request.json
#    data = request.json
#    model_name = data.get('model_name')
#    input_data = data.get('input_data')
#
#    # Use the JSON data in your logic
#    return jsonify({
#        "message": f"Prediction with model: {model_name}",
#        "input_data": input_data
#    })


if __name__ == '__main__':
    app.run(port=5000)