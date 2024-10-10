import configparser
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import os

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import generate_embeddings
from rag_utils import retrieve_k_relevant_resources, prompt_formatter

app = Flask(__name__)

config = configparser.ConfigParser()
config.read('app.conf')

env = os.environ.get('FLASK_ENV', 'default')
app.config['LLM_ID'] = config[env]['LLM_ID']
app.config['EMBEDDING_MODEL_ID'] = config[env]['EMBEDDING_MODEL_ID']
app.config['EMBEDDINGS_CSV'] = config[env]['EMBEDDINGS_CSV']
app.config['N_CONTEXT_ITEMS'] = int(config[env]['N_CONTEXT_ITEMS'])

# Load the model once at app startup
embedding_model = None
model = None

# Load the models at startup
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", quantization_config=quantization_config)
#
embedding_model = generate_embeddings.load_model(generate_embeddings.MODEL_ID, device="cuda")
print("Finished loading the models..")

# load embeddings
embeddings_df = pd.read_csv(app.config['EMBEDDINGS_CSV'])
embeddings_df['embedding'] = embeddings_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), dtype=np.float32, sep=" "))
embeddings_df = embeddings_df[embeddings_df['sentence_type']=="description"]
embeddings = np.array(embeddings_df["embedding"].tolist())


def inference_prompt(input_text):
    dialogue_template = [
        {
            "role": "user",
            "content": input_text
        }
    ]

    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                        tokenize=False,
                                        add_generation_prompt=True)
    
    return prompt


@app.route('/get_embeddings', methods=['POST'])
def get_embeddings():
    data = request.json
    query = data.get('query')
    embeddings = generate_embeddings.get_embeddings(embedding_model, query)

    # Use the JSON data in your logic
    return jsonify({
        "query": f"Query: {query}",
        "embeddings": f"embeddings: {embeddings}"
    })

@app.route('/get_relevant_resources', methods=['POST'])
def get_relevant_resources():
    data = request.json
    query = data.get('query')
    query_emb = generate_embeddings.get_embeddings(embedding_model, query)

    scores, indices = retrieve_k_relevant_resources(query_emb, embeddings, app.config['N_CONTEXT_ITEMS'])

    context_items = embeddings_df.iloc[[index.item() for index in indices]]['sentence'].tolist()

    # Use the JSON data in your logic
    return jsonify({
        "context_items": context_items,
        "scores": f"scores {scores}",
    })

@app.route('/prompt_augmentation', methods=['POST'])
def prompt_augmentation():
    data = request.json
    query = data.get('query')
    context_items = data.get('context_items')
 
    prompt_augmented = prompt_formatter(query, context_items)

    return jsonify({
        "prompt_augmented": f"{prompt_augmented}",
    })

@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    regular_query = data.get('regular_query')
    augmented_query = data.get('augmented_query')

    regular_inference_prompt = inference_prompt(regular_query)
    augmented_inference_prompt = inference_prompt(augmented_query)

    # Tokenize the input text (turn it into numbers) and send it to GPU
    regular_input_ids = tokenizer(regular_inference_prompt, return_tensors="pt").to("cuda")
    augmented_input_ids = tokenizer(augmented_inference_prompt, return_tensors="pt").to("cuda")

    # Generate outputs passed on the tokenized input
    # See generate docs: https://huggingface.co/docs/transformers/v4.38.2/en/main_classes/text_generation#transformers.GenerationConfig 
    regular_outputs = model.generate(**regular_input_ids,
                            max_new_tokens=256) # define the maximum number of new tokens to create

    augmented_outputs = model.generate(**augmented_input_ids,
                            max_new_tokens=256) # define the maximum number of new tokens to create

    regular_outputs_decoded = tokenizer.decode(regular_outputs[0])
    augmented_outputs_decoded = tokenizer.decode(augmented_outputs[0])

    return jsonify({
        "regular_decoded_output": f"{regular_outputs_decoded}",
        "augmented_decoded_output": f"{augmented_outputs_decoded}",
    })


if __name__ == '__main__':
    app.run(port=5000)