import argparse
from dotenv import load_dotenv
import json
from openai import OpenAI
import os
from tqdm import tqdm

from utils import str2bool

load_dotenv()
client = OpenAI()

def craft_translation_prompt(input_text):
    messages=[
        {
        "role": "system",
        "content": "You are a helpful assistant that is responsible for translating text from portuguese to english if the text is in portuguese. \
The text may contain named entities such as persons or companies so this should not be translated. You should also remove emojis or hyperlinks if they appear in the text. \
The response you give must contain only the processed text, no need for additional explanations."
        },
        {
        "role": "user",
        "content": f"Process and translate the following text: {input_text}"
        }]

    return messages

def chatgpt_inference(messages):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return completion.choices[0].message


def additional_preprocessing(text, context=""):
    """Adds episode context and does some basic preprocessing"""

    processed_text = text.replace("\n", " ")
    processed_text = " ".join(processed_text.split())
    processed_text = context + processed_text

    return processed_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=str, help="Input file containing all the raw data extracted from Spotify.")
    parser.add_argument("output_dir", type=str, help="Output directory to save the processed episodes data.")
    parser.add_argument("--overwrite", type=str2bool, default=False, help="If set to True will overwrite output directory if exists")

    args = parser.parse_args().__dict__
    input_path = args.pop("input")
    output_dir = args.pop("output_dir")
    overwrite = args.pop("overwrite")

    with open(input_path) as input_fp:
        raw_data = json.load(input_fp)

    total_episodes = raw_data['episodes']['total']
    processed_data = {}

    for i, episode_data in tqdm(enumerate(raw_data['episodes']['items'])):
        processed_episode = {}
        id = total_episodes-i

        name_prompt = craft_translation_prompt(episode_data['name'])
        name_result = chatgpt_inference(name_prompt)
        processed_episode['name'] = name_result.content

        description_prompt = craft_translation_prompt(episode_data['description'])
        description_result = chatgpt_inference(description_prompt)
        processed_episode['description'] = description_result.content

        processed_data[id] = processed_episode

    for id, processed_episode in processed_data.items():
        context_name = f"Episode {id} name: "
        processed_data[id]['name'] = additional_preprocessing(processed_data[id]['name'], context_name)

        context_desc = f"Episode {id} description: "
        processed_data[id]['description'] = additional_preprocessing(processed_data[id]['description'], context_desc)

    # TODO -> repeated code in 00_data_collection.py
    if os.path.exists(output_dir) and not overwrite:
        raise ValueError("Specified `output_dir` already exists. Either change `output_dir` or set `overwrite` to True.")

    output_dir = os.path.join(output_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)

    output_json_path = os.path.join(output_dir, "episodes_data.json")
    with open(output_json_path, "w") as output_fp:
        json.dump(processed_data, output_fp, ensure_ascii=False, indent=2)