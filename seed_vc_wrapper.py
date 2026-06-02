import torch
import torchaudio
import librosa
import numpy as np
from pydub import AudioSegment
import yaml
from modules.commons import build_model, load_checkpoint, recursive_munch
from hf_utils import load_custom_model_from_hf
from modules.campplus.DTDNN import CAMPPlus
from modules.bigvgan import bigvgan
from modules.audio import mel_spectrogram
from modules.rmvpe import RMVPE
from transformers import AutoFeatureExtractor, WhisperModel

class SeedVCWrapper:
    def __init__(self, device=None):
        """
        Initialize the Seed-VC wrapper with all necessary models and configurations.
        
        Args:
            device: torch device to use. If None, will be automatically determined.
        """
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        # Load base model and configuration
        self._load_base_model()
        
        # Load F0 conditioned model
        self._load_f0_model()
        
        # Load additional modules
        self._load_additional_modules()
        
        # Set streaming parameters
        self.overlap_frame_len = 16
        self.bitrate = "320k"
        
    def _load_base_model(self):
        """Load the base DiT model for voice conversion."""
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
            "Plachta/Seed-VC",
            "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
            "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
        )
        config = yaml.safe_load(open(dit_config_path, 'r'))
        model_params = recursive_munch(config['model_params'])
        self.model = build_model(model_params, stage='DiT')
        self.hop_length = config['preprocess_params']['spect_params']['hop_length']
        self.sr = config['preprocess_params']['sr']
        
        # Load checkpoints
        self.model, _, _, _ = load_checkpoint(
            self.model, None, dit_checkpoint_path,
            load_only_params=True, ignore_modules=[], is_distributed=False
        )
        for key in self.model:
            self.model[key].eval()
            self.model[key].to(self.device)
        self.model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        
        # Set up mel spectrogram function
        mel_fn_args = {
            "n_fft": config['preprocess_params']['spect_params']['n_fft'],
            "win_size": config['preprocess_params']['spect_params']['win_length'],
            "hop_size": config['preprocess_params']['spect_params']['hop_length'],
            "num_mels": config['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.sr,
            "fmin": 0,
            "fmax": None,
            "center": False
        }
        self.to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)
        
        # Load whisper model
        whisper_name = model_params.speech_tokenizer.whisper_name if hasattr(model_params.speech_tokenizer, 'whisper_name') else "openai/whisper-small"
        self.whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(self.device)
        del self.whisper_model.decoder
        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)
        
    def _load_f0_model(self):
        """Load the F0 conditioned model for voice conversion."""
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
            "Plachta/Seed-VC",
            "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
            "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
        )
        config = yaml.safe_load(open(dit_config_path, 'r'))
        model_params = recursive_munch(config['model_params'])
        self.model_f0 = build_model(model_params, stage='DiT')
        self.hop_length_f0 = config['preprocess_params']['spect_params']['hop_length']
        self.sr_f0 = config['preprocess_params']['sr']
        
        # Load checkpoints
        self.model_f0, _, _, _ = load_checkpoint(
            self.model_f0, None, dit_checkpoint_path,
            load_only_params=True, ignore_modules=[], is_distributed=False
        )
        for key in self.model_f0:
            self.model_f0[key].eval()
            self.model_f0[key].to(self.device)
        self.model_f0.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        
        # Set up mel spectrogram function for F0 model
        mel_fn_args_f0 = {
            "n_fft": config['preprocess_params']['spect_params']['n_fft'],
            "win_size": config['preprocess_params']['spect_params']['win_length'],
            "hop_size": config['preprocess_params']['spect_params']['hop_length'],
            "num_mels": config['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.sr_f0,
            "fmin": 0,
            "fmax": None,
            "center": False
        }
        self.to_mel_f0 = lambda x: mel_spectrogram(x, **mel_fn_args_f0)
        
    def _load_additional_modules(self):
        """Load additional modules like CAMPPlus, BigVGAN, and RMVPE."""
        # Load CAMPPlus
        campplus_ckpt_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model.eval()
        self.campplus_model.to(self.device)
        
        # Load BigVGAN models
        self.bigvgan_model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_22khz_80band_256x', use_cuda_kernel=False)
        self.bigvgan_model.remove_weight_norm()
        self.bigvgan_model = self.bigvgan_model.eval().to(self.device)
        
        self.bigvgan_44k_model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_512x', use_cuda_kernel=False)
        self.bigvgan_44k_model.remove_weight_norm()
        self.bigvgan_44k_model = self.bigvgan_44k_model.eval().to(self.device)
        
        # Load RMVPE for F0 extraction
        model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
        self.rmvpe = RMVPE(model_path, is_half=False, device=self.device)
        
    @staticmethod
    def adjust_f0_semitones(f0_sequence, n_semitones):
        """Adjust F0 values by a number of semitones."""
        factor = 2 ** (n_semitones / 12)
        return f0_sequence * factor
    
    @staticmethod
    def crossfade(chunk1, chunk2, overlap):
        """Apply crossfade between two audio chunks."""
        fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
        fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
        if len(chunk2) < overlap:
            chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
        else:
            chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
        return chunk2
    
    def _stream_wave_chunks(self, vc_wave, processed_frames, vc_target, overlap_wave_len, 
                           generated_wave_chunks, previous_chunk, is_last_chunk, stream_output, sr):
        """
        Helper method to handle streaming wave chunks.
        
        Args:
            vc_wave: The current wave chunk
            processed_frames: Number of frames processed so far
            vc_target: The target mel spectrogram
            overlap_wave_len: Length of overlap between chunks
            generated_wave_chunks: List of generated wave chunks
            previous_chunk: Previous wave chunk for crossfading
            is_last_chunk: Whether this is the last chunk
            stream_output: Whether to stream the output
            sr: Sample rate
            
        Returns:
            Tuple of (processed_frames, previous_chunk, should_break, mp3_bytes, full_audio)
            where should_break indicates if processing should stop
            mp3_bytes is the MP3 bytes if streaming, None otherwise
            full_audio is the full audio if this is the last chunk, None otherwise
        """
        mp3_bytes = None
        full_audio = None
        
        if processed_frames == 0:
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                
                if stream_output:
                    output_wave_int16 = (output_wave * 32768.0).astype(np.int16)
                    mp3_bytes = AudioSegment(
                        output_wave_int16.tobytes(), frame_rate=sr,
                        sample_width=output_wave_int16.dtype.itemsize, channels=1
                    ).export(format="mp3", bitrate=self.bitrate).read()
                    full_audio = (sr, np.concatenate(generated_wave_chunks))
                else:
                    return processed_frames, previous_chunk, True, None, np.concatenate(generated_wave_chunks)
                
                return processed_frames, previous_chunk, True, mp3_bytes, full_audio
            
            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - self.overlap_frame_len
            
            if stream_output:
                output_wave_int16 = (output_wave * 32768.0).astype(np.int16)
                mp3_bytes = AudioSegment(
                    output_wave_int16.tobytes(), frame_rate=sr,
                    sample_width=output_wave_int16.dtype.itemsize, channels=1
                ).export(format="mp3", bitrate=self.bitrate).read()
            
        elif is_last_chunk:
            output_wave = self.crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_target.size(2) - self.overlap_frame_len
            
            if stream_output:
                output_wave_int16 = (output_wave * 32768.0).astype(np.int16)
                mp3_bytes = AudioSegment(
                    output_wave_int16.tobytes(), frame_rate=sr,
                    sample_width=output_wave_int16.dtype.itemsize, channels=1
                ).export(format="mp3", bitrate=self.bitrate).read()
                full_audio = (sr, np.concatenate(generated_wave_chunks))
            else:
                return processed_frames, previous_chunk, True, None, np.concatenate(generated_wave_chunks)
            
            return processed_frames, previous_chunk, True, mp3_bytes, full_audio
            
        else:
            output_wave = self.crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - self.overlap_frame_len
            
            if stream_output:
                output_wave_int16 = (output_wave * 32768.0).astype(np.int16)
                mp3_bytes = AudioSegment(
                    output_wave_int16.tobytes(), frame_rate=sr,
                    sample_width=output_wave_int16.dtype.itemsize, channels=1
                ).export(format="mp3", bitrate=self.bitrate).read()
                
        return processed_frames, previous_chunk, False, mp3_bytes, full_audio

    def _process_whisper_features(self, audio_16k, is_source=True):
        """Process audio through Whisper model to extract features."""
        if audio_16k.size(-1) <= 16000 * 30:
            # If audio is short enough, process in one go
            inputs = self.whisper_feature_extractor(
                [audio_16k.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=16000
            )
            input_features = self.whisper_model._mask_input_features(
                inputs.input_features, attention_mask=inputs.attention_mask
            ).to(self.device)
            outputs = self.whisper_model.encoder(
                input_features.to(self.whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            features = outputs.last_hidden_state.to(torch.float32)
            features = features[:, :audio_16k.size(-1) // 320 + 1]
        else:
            # Process long audio in chunks
            overlapping_time = 5  # 5 seconds
            features_list = []
            buffer = None
            traversed_time = 0
            while traversed_time < audio_16k.size(-1):
                if buffer is None:  # first chunk
                    chunk = audio_16k[:, traversed_time:traversed_time + 16000 * 30]
                else:
                    chunk = torch.cat([
                        buffer, 
                        audio_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]
                    ], dim=-1)
                inputs = self.whisper_feature_extractor(
                    [chunk.squeeze(0).cpu().numpy()],
                    return_tensors="pt",
                    return_attention_mask=True,
                    sampling_rate=16000
                )
                input_features = self.whisper_model._mask_input_features(
                    inputs.input_features, attention_mask=inputs.attention_mask
                ).to(self.device)
                outputs = self.whisper_model.encoder(
                    input_features.to(self.whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                chunk_features = outputs.last_hidden_state.to(torch.float32)
                chunk_features = chunk_features[:, :chunk.size(-1) // 320 + 1]
                if traversed_time == 0:
                    features_list.append(chunk_features)
                else:
                    features_list.append(chunk_features[:, 50 * overlapping_time:])
                buffer = chunk[:, -16000 * overlapping_time:]
                traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
            features = torch.cat(features_list, dim=1)
        
        return features
    @torch.no_grad()
    @torch.inference_mode()
    def convert_voice(
        self,
        source,
        target,
        diffusion_steps=50,
        length_adjust=1.0,
        inference_cfg_rate=0.9,
        f0_condition=True,
        auto_f0_adjust=True,
        pitch_shift=0,
        stream_output=False,
        expressive_embeddings=None
    ):

        # --------------------------------------------------------
        # Select models
        # --------------------------------------------------------

        inference_module = (
            self.model
            if not f0_condition
            else self.model_f0
        )

        mel_fn = (
            self.to_mel
            if not f0_condition
            else self.to_mel_f0
        )

        bigvgan_fn = (
            self.bigvgan_model
            if not f0_condition
            else self.bigvgan_44k_model
        )

        sr = 22050 if not f0_condition else 44100

        hop_length = 256 if not f0_condition else 512

        max_context_window = (
            sr // hop_length * 30
        )

        overlap_wave_len = (
            self.overlap_frame_len
            * hop_length
        )

        # --------------------------------------------------------
        # Load audio
        # --------------------------------------------------------

        source_audio = librosa.load(
            source,
            sr=sr
        )[0]

        ref_audio = librosa.load(
            target,
            sr=sr
        )[0]

        source_audio = (
            torch.tensor(source_audio)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        ref_audio = (
            torch.tensor(ref_audio[:sr * 25])
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        # --------------------------------------------------------
        # Resample to 16k
        # --------------------------------------------------------

        ref_waves_16k = (
            torchaudio.functional.resample(
                ref_audio,
                sr,
                16000
            )
        )

        converted_waves_16k = (
            torchaudio.functional.resample(
                source_audio,
                sr,
                16000
            )
        )

        # --------------------------------------------------------
        # Whisper features
        # --------------------------------------------------------

        S_alt = self._process_whisper_features(
            converted_waves_16k,
            is_source=True
        )

        S_ori = self._process_whisper_features(
            ref_waves_16k,
            is_source=False
        )

        # --------------------------------------------------------
        # Mel spectrograms
        # --------------------------------------------------------

        mel = mel_fn(
            source_audio.float()
        )

        mel2 = mel_fn(
            ref_audio.float()
        )

        target_lengths = torch.LongTensor([
            int(mel.size(2) * length_adjust)
        ]).to(mel.device)

        target2_lengths = torch.LongTensor([
            mel2.size(2)
        ]).to(mel2.device)

        # --------------------------------------------------------
        # Style features
        # --------------------------------------------------------

        feat2 = torchaudio.compliance.kaldi.fbank(

            ref_waves_16k,

            num_mel_bins=80,

            dither=0,

            sample_frequency=16000
        )

        feat2 = (
            feat2
            - feat2.mean(dim=0, keepdim=True)
        )

        style2 = self.campplus_model(
            feat2.unsqueeze(0)
        )

        # --------------------------------------------------------
        # F0 conditioning
        # --------------------------------------------------------

        if f0_condition:

            F0_ori = self.rmvpe.infer_from_audio(
                ref_waves_16k[0],
                thred=0.03
            )

            F0_alt = self.rmvpe.infer_from_audio(
                converted_waves_16k[0],
                thred=0.03
            )

            F0_ori = (
                torch.from_numpy(F0_ori)
                .float()
                .to(self.device)[None]
            )

            F0_alt = (
                torch.from_numpy(F0_alt)
                .float()
                .to(self.device)[None]
            )

            voiced_F0_ori = F0_ori[F0_ori > 1]

            voiced_F0_alt = F0_alt[F0_alt > 1]

            log_f0_alt = torch.log(
                F0_alt + 1e-5
            )

            voiced_log_f0_ori = torch.log(
                voiced_F0_ori + 1e-5
            )

            voiced_log_f0_alt = torch.log(
                voiced_F0_alt + 1e-5
            )

            median_log_f0_ori = torch.median(
                voiced_log_f0_ori
            )

            median_log_f0_alt = torch.median(
                voiced_log_f0_alt
            )

            shifted_log_f0_alt = (
                log_f0_alt.clone()
            )

            if auto_f0_adjust:

                shifted_log_f0_alt[
                    F0_alt > 1
                ] = (
                    log_f0_alt[F0_alt > 1]
                    - median_log_f0_alt
                    + median_log_f0_ori
                )

            shifted_f0_alt = torch.exp(
                shifted_log_f0_alt
            )

            if pitch_shift != 0:

                shifted_f0_alt[
                    F0_alt > 1
                ] = self.adjust_f0_semitones(

                    shifted_f0_alt[F0_alt > 1],

                    pitch_shift
                )

        else:

            F0_ori = None
            shifted_f0_alt = None

        # --------------------------------------------------------
        # Length regulation
        # --------------------------------------------------------

        cond, _, _, _, _ = (
            inference_module.length_regulator(

                S_alt,

                ylens=target_lengths,

                n_quantizers=3,

                f0=shifted_f0_alt
            )
        )

        prompt_condition, _, _, _, _ = (
            inference_module.length_regulator(

                S_ori,

                ylens=target2_lengths,

                n_quantizers=3,

                f0=F0_ori
            )
        )

        # --------------------------------------------------------
        # Expressive embedding injection
        # --------------------------------------------------------

        if expressive_embeddings is not None:

            try:

                print(
                    "[SeedVC] Injecting expressive embeddings..."
                )

                expressive_embeddings = (
                    np.asarray(
                        expressive_embeddings,
                        dtype=np.float32
                    )
                )

                expressive_embeddings = (
                    torch.tensor(
                        expressive_embeddings,
                        device=self.device,
                        dtype=cond.dtype
                    )
                )

                target_len = cond.shape[1]

                if expressive_embeddings.shape[0] != target_len:

                    expressive_embeddings = (
                        torch.nn.functional.interpolate(

                            expressive_embeddings.T.unsqueeze(0),

                            size=target_len,

                            mode="linear",

                            align_corners=False

                        )
                        .squeeze(0)
                        .T
                    )

                cond_dim = cond.shape[-1]

                if expressive_embeddings.shape[-1] > cond_dim:

                    expressive_embeddings = (
                        expressive_embeddings[:, :cond_dim]
                    )

                elif expressive_embeddings.shape[-1] < cond_dim:

                    pad_size = (
                        cond_dim
                        - expressive_embeddings.shape[-1]
                    )

                    expressive_embeddings = (
                        torch.nn.functional.pad(

                            expressive_embeddings,

                            (0, pad_size)
                        )
                    )

                cond = cond + (
                    0.12
                    * expressive_embeddings.unsqueeze(0)
                )

                print(
                    "[SeedVC] Expressive conditioning injected"
                )

            except Exception as e:

                print(
                    f"[SeedVC] Injection failed: {e}"
                )

        # --------------------------------------------------------
        # Full conditioning
        # --------------------------------------------------------

        cat_condition = torch.cat(

            [prompt_condition, cond],

            dim=1
        )

        # --------------------------------------------------------
        # Diffusion inference
        # --------------------------------------------------------

        with torch.autocast(

            device_type=self.device.type,

            dtype=torch.float16
        ):

            vc_target = (
                inference_module.cfm.inference(

                    cat_condition,

                    torch.LongTensor([
                        cat_condition.size(1)
                    ]).to(mel2.device),

                    mel2,

                    style2,

                    None,

                    diffusion_steps,

                    inference_cfg_rate=
                    inference_cfg_rate
                )
            )

            vc_target = vc_target[
                :,
                :,
                mel2.size(-1):
            ]

        # --------------------------------------------------------
        # Vocoding
        # --------------------------------------------------------

        vc_wave = (
            bigvgan_fn(
                vc_target.float()
            )[0]
        )

        output_wave = (
            vc_wave.cpu().numpy()
        )

        # --------------------------------------------------------
        # Final output handling
        # --------------------------------------------------------

        output_wave = np.asarray(
            output_wave,
            dtype=np.float32
        )

        # Store internally for recovery
        self.last_output_audio = output_wave

        # --------------------------------------------------------
        # Streaming mode
        # --------------------------------------------------------

        if stream_output:

            def generator():

                yield None, (sr, output_wave)

            return generator()

        # --------------------------------------------------------
        # Normal mode
        # --------------------------------------------------------

        return output_wave