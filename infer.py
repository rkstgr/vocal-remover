import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from mdx import MDX

PROJECT_ROOT = Path(__file__).parent


# main cli entry point

model_params = {
    "compensate": 1.043,
    }


def get_model_params(model_path) -> dict:
    model_hash = hashlib.md5(open(model_path, 'rb').read()).hexdigest()
    with open(PROJECT_ROOT / "models/MDX_NET_Models/model_data/model_data.json", "r") as f:
        all_model_settings = json.load(f)
        print(all_model_settings)
    return all_model_settings[model_hash]


def process(input_paths):
    """Start the conversion for all the given mp3 and wav files
    
    input_paths: list of paths to the audio files
    """
    for path in input_paths:
        model_path = PROJECT_ROOT / "models/MDX_NET_Models/Kim_Vocal_1.onnx"
        model = MDX(model_path, {}, device="cpu")
        model.separate_file(path, primary_stem=True, secondary_stem=False)


def main():
    input_paths = ["/Users/erik/Downloads/Nina Chuba/Lights out sample.mp3"]
    process(input_paths)


def test_mdx():
    input_paths = ["/Users/erik/Downloads/Nina Chuba/Lights out sample.mp3"]
    process(input_paths)


if __name__ == '__main__':
    main()
