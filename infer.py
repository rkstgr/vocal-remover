import argparse
import dataclasses
import hashlib
import json
from dataclasses import dataclass
import time
from pathlib import Path
import glob
from tqdm.auto import tqdm
import torch.cuda

from mdx import MDX

PROJECT_ROOT = Path(__file__).parent

AUDIO_EXTENSIONS = [".mp3", ".wav", ".ogg", ".flac", ".opus"]

def get_audio_files(input_path, recursive):
    """Get a list of all the audio files in the given input path"""
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        search_pattern = "**/*" if recursive else "*"
        files = []
        for ext in AUDIO_EXTENSIONS:
            files.extend(input_path.glob(f"{search_pattern}{ext}"))
        return files
    else:
        raise ValueError("Input path is not a valid file or directory.")


def process(input_paths, output_dir=None, recursive=False, device="cpu"):
    """Start the conversion for all the given mp3 and wav files

    input_paths: list of paths to the audio files
    """
    print(f"Using device: {device}")
    model_path = PROJECT_ROOT / "models/MDX_Net_Models/Kim_Vocal_1.onnx"
    model = MDX(model_path, {}, device=device)

    # Wrap the outer loop with a tqdm progress bar
    for input_path in tqdm(input_paths, desc="Processing files", unit="file"):
        if output_dir:
            output_path = output_dir
        else:
            output_path = input_path.parent

        # Wrap the model's separate_file method to show a tqdm progress bar for each file
        def progress_callback(progress):
            # progress between 0 and 1
            progress = int(progress * 100)
            pbar.update(progress - pbar.n)

        pbar = tqdm(total=100, desc=f"{input_path.name}", unit="%")
        model.separate_file(input_path, output_path, primary_stem=True, secondary_stem=False,
                            progress_callback=progress_callback)
        pbar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input path (file or directory)")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--recursive", action="store_true", help="Recursively process directories")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on. Defaults to 'cuda' if available, otherwise 'cpu'")

    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    if output_dir and not output_dir.is_dir():
        raise ValueError("Output directory does not exist or is not a directory.")

    input_paths = get_audio_files(input_path, args.recursive)

    start = time.time()
    process(input_paths, output_dir, args.recursive, args.device)
    end = time.time()
    print(f"Total time elapsed: {end - start:.2f} seconds")


if __name__ == '__main__':
    main()
