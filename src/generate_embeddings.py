import argparse
import json
import numpy as np
import os
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from utils import save_output_file, str2bool

MODEL_ID = "all-mpnet-base-v2"

def load_model(model_id, device=None):
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(model_name_or_path=model_id, device=device)
    return embedding_model

def get_embeddings(embedding_model, text):
    return embedding_model.encode(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=str, help="JSON containing processed podcast episodes names and descriptions.")
    parser.add_argument("output_dir", type=str, help="Output directory to save the embeddings.")
    parser.add_argument("--device", type=str, default="cuda", help="Select device on which to run the embedding model.")
    parser.add_argument("--overwrite", type=str2bool, default=False, help="If set to True will overwrite output directory if exists")

    args = parser.parse_args().__dict__
    input_path = args.pop("input")
    output_dir = args.pop("output_dir")
    device = args.pop("device")
    overwrite = args.pop("overwrite")

    with open(input_path) as input_fp:
        processed_data = json.load(input_fp)

    embedding_model = load_model(MODEL_ID, device=device)
    embeddings = []
    embeddings_mapping = {}
    embeddings = {}

    import pandas as pd
    embeddings_list = []
    sentences_list = []
    sentence_type_list = []
    episode_ids_list = []
    for id, episode in tqdm(processed_data.items()):
        name_emb = get_embeddings(embedding_model, episode['name'])
        sentence_type_list.append('name')
        sentences_list.append(episode['name'])
        embeddings_list.append(name_emb)
        episode_ids_list.append(id)

        desc_emb = get_embeddings(embedding_model, episode['description'])
        sentence_type_list.append('description')
        sentences_list.append(episode['description'])
        embeddings_list.append(desc_emb)
        episode_ids_list.append(id)

    embeddings_df = pd.DataFrame(data=zip(sentences_list, embeddings_list, sentence_type_list, episode_ids_list),
                                 columns=["sentence", "embedding", "sentence_type", "episode_id"])

    save_output_file(embeddings_df, output_dir, "embeddings", "episode_embeddings.csv", overwrite)