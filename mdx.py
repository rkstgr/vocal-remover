import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Callable

import librosa
import torch
import numpy as np

from lib_v5 import mdxnet
from lib_v5 import spec_utils
import onnxruntime as ort
import soundfile as sf

from separate import rerun_mp3


def write_audio(stem_path, stem_audio, sample_rate):
    sf.write(str(stem_path), stem_audio, sample_rate)
    # encode(stem_path, target_encoding)


def get_audio_codec(file_path: Path) -> str:
    cmd = f"ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 \"{str(file_path)}\""
    codec = subprocess.check_output(cmd, shell=True, text=True).strip()
    return codec


def encode(target_audio_file: Path, original_file: Path):
    codec = get_audio_codec(original_file)
    output_file = target_audio_file.parent / (target_audio_file.stem + "_encoded" + original_file.suffix)

    cmd = f"ffmpeg -i \"{target_audio_file}\" -strict -2 -vn -y -c:a {codec} \"{output_file}\""
    subprocess.run(cmd, shell=True)

    return output_file


@NotImplementedError
@dataclass
# target_name, lr, optimizer, dim_c, dim_f, dim_t, n_fft, hop_length,
#                  num_blocks, l, g, k, bn, bias, overlap
class MDXModelParams:
    target_name: str
    lr: float
    optimizer: str
    dim_c: int
    dim_f: int
    dim_t: int
    n_fft: int
    hop_length: int
    num_blocks: int
    l: int
    g: int
    k: int
    bn: bool
    bias: bool
    overlap: int


class KimVocal1Params:
    n_fft = 7680
    hop = 1024
    dim_t = 1024
    dim_c = 4
    n_bins = 3841
    dim_f = 3072


def prepare_mix(mix, chunk_set, margin_set, mdx_net_cut=False, sample_rate=44100):
    mix_path = mix
    if isinstance(mix, str):
        if not Path(mix).exists():
            raise FileNotFoundError(f'File {mix_path} does not exist')

        mix, _ = librosa.load(mix_path, mono=False, sr=sample_rate)

    if not np.any(mix) and mix_path.endswith('.mp3'):
        mix = rerun_mp3(mix_path)

    if mix.ndim == 1:
        mix = np.asfortranarray([mix, mix])
    else:
        # if [n, 2] -> [2, n]
        if mix.shape[1] in [1, 2]:
            mix = np.asfortranarray(mix.T)

    def get_segmented_mix(chunk_set=chunk_set):
        segmented_mix = {}

        samples = mix.shape[-1]
        margin = margin_set
        chunk_size = chunk_set * sample_rate
        assert not margin == 0, 'margin cannot be zero!'

        if margin > chunk_size:
            margin = chunk_size
        if chunk_set == 0 or samples < chunk_size:
            chunk_size = samples

        counter = -1
        for skip in range(0, samples, chunk_size):
            counter += 1
            s_margin = 0 if counter == 0 else margin
            end = min(skip + chunk_size + margin, samples)
            start = skip - s_margin
            segmented_mix[skip] = mix[:, start:end].copy()
            if end == samples:
                break

        return segmented_mix

    segmented_mix = get_segmented_mix()
    raw_mix = get_segmented_mix(chunk_set=0) if mdx_net_cut else mix
    return segmented_mix, raw_mix, sample_rate


class MDX:
    SAMPLE_RATE = 44100

    def __init__(self, model_path: Union[str, Path], model_params: dict, device: str = 'cuda'):
        self.model_path = model_path
        self.device = device
        self.model_params = self.initialize_model_params()
        self.model_fn = self.initialize_model_fn(self.model_path, self.device)

        self.chunks = 0
        self.margin = self.SAMPLE_RATE

    @staticmethod
    def initialize_model_fn(model_path, device) -> Callable[[torch.Tensor], torch.Tensor]:
        if model_path.suffix == '.ckpt':
            model_params = torch.load(str(model_path), map_location=device)['hyper_parameters']
            separator = mdxnet.ConvTDFNet(**model_params)
            return separator.load_from_checkpoint(model_path).to(device).eval()
        elif model_path.suffix == '.onnx':
            if device == "cuda":
                ort_providers = ['CUDAExecutionProvider']
            else:
                ort_providers = ['CPUExecutionProvider']
            ort_ = ort.InferenceSession(str(model_path), providers=ort_providers)
            return lambda ort_input: ort_.run(None, {'input': ort_input.cpu().numpy()})[0]
        else:
            raise ValueError('Invalid model format. Please use either .ckpt or .onnx')

    def separate_file(self, file_path, output_dir, primary_stem=True, secondary_stem=False, progress_callback=None):

        audio_file_path = Path(file_path)

        primary_stem_label = "vocals"
        secondary_stem_label = "instrumental"
        normalize = False

        # mix, raw_mix are dicts
        mix, raw_mix, samplerate = prepare_mix(str(audio_file_path), self.chunks, self.margin, mdx_net_cut=True,
                                               sample_rate=self.SAMPLE_RATE)
        source = self.demix_base(mix, is_ckpt="ckpt" in self.model_path.suffix, progress_callback=progress_callback)[0]

        if primary_stem:
            # use audio_file_path name
            primary_stem_path = Path(output_dir) / f'{audio_file_path.stem}_({primary_stem_label}).wav'
            primary_source = spec_utils.normalize(source, normalize).T

            write_audio(primary_stem_path, primary_source, self.SAMPLE_RATE)
            encode(primary_stem_path, original_file=audio_file_path)
            os.unlink(primary_stem_path)

        if secondary_stem:
            secondary_stem_path = Path(output_dir) / f'{audio_file_path.stem}_({secondary_stem_label}).wav'
            raw_mix = self.demix_base(raw_mix, is_match_mix=True)[0]
            self.secondary_source, raw_mix = spec_utils.normalize_two_stem(source * self.model_params["compensate"],
                                                                           raw_mix, False)

            self.secondary_source = (-self.secondary_source.T + raw_mix.T)

            write_audio(secondary_stem_path, self.secondary_source, self.SAMPLE_RATE)
            encode(secondary_stem_path, original_file=audio_file_path)
            os.unlink(secondary_stem_path)

    def initialize_model_params(self):
        """Initialize the model settings"""

        # hardcoded for mdx model "kim vocal 1"
        n_fft = 7680
        hop = 1024
        dim_t = 256
        dim_c = 4
        n_bins = 3841
        dim_f = 3072

        params = {
            "n_fft": n_fft,
            "hop": hop,
            "dim_t": dim_t,
            "dim_c": dim_c,
            "dim_f": dim_f,
            "n_bins": n_fft // 2 + 1,
            "trim": n_fft // 2,
            "chunk_size": hop * (dim_t - 1),
            "window": torch.hann_window(window_length=n_fft, periodic=False).to(self.device),
            "freq_pad": torch.zeros([1, dim_c, n_bins - dim_f, dim_t]).to(self.device),
            "gen_size": hop * (dim_t - 1) - 2 * (n_fft // 2),
        }
        return params

    def initialize_mix(self, mix, is_ckpt=False):
        gen_size = self.model_params["gen_size"]
        trim = self.model_params["trim"]
        chunk_size = self.model_params["chunk_size"]

        if is_ckpt:
            pad = gen_size + trim - ((mix.shape[-1]) % gen_size)
            mixture = np.concatenate(
                (np.zeros((2, trim), dtype='float32'), mix, np.zeros((2, pad), dtype='float32')), 1)
            num_chunks = mixture.shape[-1] // gen_size
            mix_waves = [mixture[:, i * gen_size: i * gen_size + chunk_size] for i in range(num_chunks)]
        else:
            mix_waves = []
            n_sample = mix.shape[1]
            pad = gen_size - n_sample % gen_size
            mix_p = np.concatenate((np.zeros((2, trim)), mix, np.zeros((2, pad)), np.zeros((2, trim))), 1)
            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i:i + chunk_size])
                mix_waves.append(waves)
                i += gen_size

        mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(self.device)

        return mix_waves, pad

    def demix_base(self, mix, is_ckpt=False, is_match_mix=False, progress_callback=None):
        trim = self.model_params["trim"]
        adjust = 1
        mdx_batch_size = 1

        chunked_sources = []
        total_mix_waves = -1
        current_mix_waves = 0
        for slice in mix:
            sources = []
            tar_waves_ = []
            mix_p = mix[slice]
            mix_waves, pad = self.initialize_mix(mix_p, is_ckpt=is_ckpt)
            mix_waves = mix_waves.split(mdx_batch_size)
            pad = mix_p.shape[-1] if is_ckpt else -pad
            if total_mix_waves == -1:
                total_mix_waves = len(mix_waves) * len(mix)
            with torch.no_grad():
                for mix_wave in mix_waves:
                    tar_waves = self.run_model(mix_wave, is_ckpt=is_ckpt, is_match_mix=is_match_mix)
                    tar_waves_.append(tar_waves)
                    current_mix_waves += 1
                    if progress_callback:
                        progress_callback(current_mix_waves / total_mix_waves)
                tar_waves_ = np.vstack(tar_waves_)[:, :, trim:-trim] if is_ckpt else tar_waves_
                tar_waves = np.concatenate(tar_waves_, axis=-1)[:, :pad]
                start = 0 if slice == 0 else self.margin
                end = None if slice == list(mix.keys())[::-1][0] or self.margin == 0 else -self.margin
                sources.append(tar_waves[:, start:end] * (1 / adjust))
            chunked_sources.append(sources)
        sources = np.concatenate(chunked_sources, axis=-1)

        return sources

    def run_model(self, mix, is_ckpt=False, is_match_mix=False):
        adjust = 1
        trim = self.model_params["trim"]

        spek = self.stft(mix.to(self.device)) * adjust
        spek[:, :, :3, :] *= 0

        if is_match_mix:
            spec_pred = spek.cpu().numpy()
        else:
            spec_pred = -self.model_fn(-spek) * 0.5 + self.model_fn(spek)

        if is_ckpt:
            return self.istft(spec_pred).cpu().detach().numpy()
        else:
            return self.istft(torch.tensor(spec_pred).to(self.device)).to("cpu")[:, :, trim:-trim].transpose(
                0, 1).reshape(2, -1).numpy()

    def stft(self, x: torch.Tensor):
        params = self.model_params
        chunk_size = params["chunk_size"]
        x = x.reshape([-1, chunk_size])
        x = torch.stft(x, n_fft=params["n_fft"], hop_length=params["hop"], window=params["window"],
                       center=True, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, params["n_bins"], params["dim_t"]]) \
            .reshape([-1, params["dim_c"], params["n_bins"], params["dim_t"]])
        return x[:, :, :params["dim_f"]]

    def istft(self, x, freq_pad=None):
        params = self.model_params
        freq_pad = params["freq_pad"].repeat([x.shape[0], 1, 1, 1]) if freq_pad is None else freq_pad
        x = torch.cat([x, freq_pad], -2)
        x = x.reshape([-1, 2, 2, params["n_bins"], params["dim_t"]]).reshape([-1, 2, params["n_bins"], params["dim_t"]])
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(x, n_fft=params["n_fft"], hop_length=params["hop"], window=params["window"],
                        center=True)
        return x.reshape([-1, 2, params["chunk_size"]])
