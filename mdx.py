import os
import torch
import numpy as np
from gui_data.constants import MDX_NET_FREQ_CUT, SAVING_STEM
from lib_v5.mdxnet import MdxnetSet
from lib_v5 import spec_utils
import onnxruntime as ort

from separate import prepare_mix, process_secondary_model

class MDX:
    def __init__(self, model_path):
        self.model_path = model_path
        # TODO load model into memory

    def write_to_console(self, text, base_text='\n'):
        print(f'{base_text}{text}', end='\r')

    # MAIN METHOD
    def seperate(self, audio_file_path):
        audio_file_path = Path(audio_file_path)
        samplerate = 44100
          
        if self.primary_model_name == self.model_basename and self.primary_sources:
            self.primary_source, self.secondary_source = self.load_cached_sources()
        else:
            self.start_inference_console_write()

        if self.model_path.endswith('.ckpt'):
            model_params = torch.load(self.model_path, map_location=lambda storage, loc: storage)['hyper_parameters']
            self.dim_c, self.hop = model_params['dim_c'], model_params['hop_length']
            separator = MdxnetSet.ConvTDFNet(**model_params)
            self.model_run = separator.load_from_checkpoint(self.model_path).to(self.device).eval()
        elif self.model_path.endswith('.onnx'):
            ort_ = ort.InferenceSession(self.model_path, providers=self.run_type)
            self.model_run = lambda spek:ort_.run(None, {'input': spek.cpu().numpy()})[0]
        else:
            raise ValueError('Invalid model format. Please use either .ckpt or .onnx')

        self.initialize_model_settings()
        self.running_inference_console_write()
        mdx_net_cut = True if self.primary_stem in MDX_NET_FREQ_CUT else False
        mix, raw_mix, samplerate = prepare_mix(self.audio_file, self.chunks, self.margin, mdx_net_cut=mdx_net_cut)
        source = self.demix_base(mix, is_ckpt=self.is_mdx_ckpt)[0]
        self.write_to_console("DONE", base_text='')            

        if self.is_secondary_model_activated:
            if self.secondary_model:
                self.secondary_source_primary, self.secondary_source_secondary = process_secondary_model(self.secondary_model, self.process_data, main_process_method=self.process_method)
        
        if not self.is_secondary_stem_only:
            self.write_to_console(f'{SAVING_STEM[0]}{self.primary_stem}{SAVING_STEM[1]}') if not self.is_secondary_model else None
            primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.primary_stem}).wav')
            if not isinstance(self.primary_source, np.ndarray):
                self.primary_source = spec_utils.normalize(source, self.is_normalization).T
            self.primary_source_map = {self.primary_stem: self.primary_source}
            self.write_audio(primary_stem_path, self.primary_source, samplerate, self.secondary_source_primary)

        if not self.is_primary_stem_only:
            self.write_to_console(f'{SAVING_STEM[0]}{self.secondary_stem}{SAVING_STEM[1]}') if not self.is_secondary_model else None
            secondary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.secondary_stem}).wav')
            if not isinstance(self.secondary_source, np.ndarray):
                raw_mix = self.demix_base(raw_mix, is_match_mix=True)[0] if mdx_net_cut else raw_mix
                self.secondary_source, raw_mix = spec_utils.normalize_two_stem(source*self.compensate, raw_mix, self.is_normalization)
            
                if self.is_invert_spec:
                    self.secondary_source = spec_utils.invert_stem(raw_mix, self.secondary_source)
                else:
                    self.secondary_source = (-self.secondary_source.T+raw_mix.T)

            self.secondary_source_map = {self.secondary_stem: self.secondary_source}
            self.write_audio(secondary_stem_path, self.secondary_source, samplerate, self.secondary_source_secondary)

        torch.cuda.empty_cache()
        secondary_sources = {**self.primary_source_map, **self.secondary_source_map}

        self.cache_source(secondary_sources)

        if self.is_secondary_model:
            return secondary_sources

    def initialize_model_settings(self):
        """Initialize the model settings"""
        self.n_bins = self.n_fft//2+1
        self.trim = self.n_fft//2
        self.chunk_size = self.hop * (self.dim_t-1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=False).to(self.device)
        self.freq_pad = torch.zeros([1, self.dim_c, self.n_bins-self.dim_f, self.dim_t]).to(self.device)
        self.gen_size = self.chunk_size-2*self.trim

    def initialize_mix(self, mix, is_ckpt=False):
        if is_ckpt:
            pad = self.gen_size + self.trim - ((mix.shape[-1]) % self.gen_size)
            mixture = np.concatenate((np.zeros((2, self.trim), dtype='float32'),mix, np.zeros((2, pad), dtype='float32')), 1)
            num_chunks = mixture.shape[-1] // self.gen_size
            mix_waves = [mixture[:, i * self.gen_size: i * self.gen_size + self.chunk_size] for i in range(num_chunks)]
        else:
            mix_waves = []
            n_sample = mix.shape[1]
            pad = self.gen_size - n_sample%self.gen_size
            mix_p = np.concatenate((np.zeros((2,self.trim)), mix, np.zeros((2,pad)), np.zeros((2,self.trim))), 1)
            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i:i+self.chunk_size])
                mix_waves.append(waves)
                i += self.gen_size
                
        mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(self.device)

        return mix_waves, pad
    
    def demix_base(self, mix, is_ckpt=False, is_match_mix=False):
        chunked_sources = []
        for slice in mix:
            sources = []
            tar_waves_ = []
            mix_p = mix[slice]
            mix_waves, pad = self.initialize_mix(mix_p, is_ckpt=is_ckpt)
            mix_waves = mix_waves.split(self.mdx_batch_size)
            pad = mix_p.shape[-1] if is_ckpt else -pad
            with torch.no_grad():
                for mix_wave in mix_waves:
                    self.running_inference_progress_bar(len(mix)*len(mix_waves), is_match_mix=is_match_mix)
                    tar_waves = self.run_model(mix_wave, is_ckpt=is_ckpt, is_match_mix=is_match_mix)
                    tar_waves_.append(tar_waves)
                tar_waves_ = np.vstack(tar_waves_)[:, :, self.trim:-self.trim] if is_ckpt else tar_waves_
                tar_waves = np.concatenate(tar_waves_, axis=-1)[:, :pad]
                start = 0 if slice == 0 else self.margin
                end = None if slice == list(mix.keys())[::-1][0] or self.margin == 0 else -self.margin
                sources.append(tar_waves[:,start:end]*(1/self.adjust))
            chunked_sources.append(sources)
        sources = np.concatenate(chunked_sources, axis=-1)
        
        return sources

    def run_model(self, mix, is_ckpt=False, is_match_mix=False):
        
        spek = self.stft(mix.to(self.device))*self.adjust
        spek[:, :, :3, :] *= 0 

        if is_match_mix:
            spec_pred = spek.cpu().numpy()
        else:
            spec_pred = -self.model_run(-spek)*0.5+self.model_run(spek)*0.5 if self.is_denoise else self.model_run(spek)

        if is_ckpt:
            return self.istft(spec_pred).cpu().detach().numpy()
        else: 
            return self.istft(torch.tensor(spec_pred).to(self.device)).to("cpu")[:,:,self.trim:-self.trim].transpose(0,1).reshape(2, -1).numpy()
    
    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True,return_complex=True)
        x=torch.view_as_real(x)
        x = x.permute([0,3,1,2])
        x = x.reshape([-1,2,2,self.n_bins,self.dim_t]).reshape([-1,self.dim_c,self.n_bins,self.dim_t])
        return x[:,:,:self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = self.freq_pad.repeat([x.shape[0],1,1,1]) if freq_pad is None else freq_pad
        x = torch.cat([x, freq_pad], -2)
        x = x.reshape([-1,2,2,self.n_bins,self.dim_t]).reshape([-1,2,self.n_bins,self.dim_t])
        x = x.permute([0,2,3,1])
        x=x.contiguous()
        x=torch.view_as_complex(x)
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        return x.reshape([-1,2,self.chunk_size])
