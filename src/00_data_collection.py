import argparse
import base64
from dotenv import load_dotenv
import json
import os
import requests

from utils import str2bool

load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

def get_token():
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "client_credentials"
    }
    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()
    response_data = response.json()
    token = response_data["access_token"]

    return token

def get_auth_header(token):
    return {"Authorization": "Bearer " + token}

def get_show_metadata(token, show_id):
    url = "https://api.spotify.com/v1/shows/"
    headers = get_auth_header(token)
    query = f"{show_id}"

    query_url = url + query
    response = requests.get(query_url, headers=headers)
    response.raise_for_status()
    response_data = response.json()

    return response_data


if __name__=="__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("show_id", type=str, help="Spotify show id")
    parser.add_argument("output_dir", type=str, help="Output directory to save the retrieved podcast episodes data.")
    parser.add_argument("--overwrite", type=str2bool, default=False, help="If set to True will overwrite output directory if exists")

    args = parser.parse_args().__dict__
    show_id = args.pop("show_id")
    output_dir = args.pop("output_dir")
    overwrite = args.pop("overwrite")

    token = get_token()

    show_metadata = get_show_metadata(token, show_id)

    if os.path.exists(output_dir) and not overwrite:
        raise ValueError("Specified `output_dir` already exists. Either change `output_dir` or set `overwrite` to True.")

    output_dir = os.path.join(output_dir, "raw")
    os.makedirs(output_dir, exist_ok=True)

    output_json_path = os.path.join(output_dir, "metadata.json")
    with open(output_json_path, "w") as fp:
        json.dump(show_metadata, fp, ensure_ascii=False, indent=2)

