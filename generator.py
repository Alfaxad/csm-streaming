from dataclasses import dataclass
import math
import os
from typing import List, Tuple, Generator as PyGenerator, Optional, Callable
import time
import queue
import threading
import platform
from typing_extensions import OrderedDict
import wave
import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import Model, ModelArgs
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)

@dataclass
class Segment:
    speaker: int
    text: str
    sample_rate = 24_000
    audio: torch.Tensor


def load_llama3_tokenizer():
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer_name = "unsloth/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )

    return tokenizer


class Generator:
    def __init__(self, model: Model):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer()
        device = next(model.parameters()).device

        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        
        num_codebooks = model.config.audio_num_codebooks
        mimi.set_num_codebooks(num_codebooks)
        self._num_codebooks = num_codebooks
        self._audio_tokenizer = mimi

        self.sample_rate = mimi.sample_rate
        self.device = device

        self._stream_buffer_size = 3  # Default for low-latency streaming
        self.max_seq_len = 2048
        self._cache = OrderedDict()
        self._text_token_cache = {}
        torch.set_num_threads(16)
        torch.cuda.set_per_process_memory_fraction(0.95)

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize text segment with caching optimization for reduced latency.
        """
        # Check cache first
        cache_key = f"{speaker}:{text}"
        if not hasattr(self, '_text_token_cache'):
            self._text_token_cache = {}
        
        if cache_key in self._text_token_cache:
            return self._text_token_cache[cache_key]

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), self._num_codebooks+1, dtype=torch.long, device=self.device)
        text_frame_mask = torch.zeros(len(text_tokens), self._num_codebooks+1, dtype=torch.bool, device=self.device)
        text_frame[:, -1] = torch.tensor(text_tokens, device=self.device)
        text_frame_mask[:, -1] = True

        frame_tokens = [text_frame]
        frame_masks = [text_frame_mask]
        
        result = (torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0))
        
        self._text_token_cache[cache_key] = result
        
        return result


    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        
        # Limit to the number of codebooks set in MIMI
        audio_tokens = audio_tokens[:self._num_codebooks, :]
        
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), self._num_codebooks+1).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), self._num_codebooks+1).bool().to(self.device)
        audio_frame[:, :self._num_codebooks] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :self._num_codebooks] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        total_len = text_tokens.size(0) + audio_tokens.size(0)

        if total_len > self.max_seq_len:
            overflow = total_len - self.max_seq_len

            if text_tokens.size(0) > overflow:
                text_tokens = text_tokens[overflow:]
                text_masks = text_masks[overflow:]
            else:
                audio_overflow = overflow - text_tokens.size(0)
                text_tokens = text_tokens[0:0]
                text_masks = text_masks[0:0]
                audio_tokens = audio_tokens[audio_overflow:]
                audio_masks = audio_masks[audio_overflow:]

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def generate_stream(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.7,
        topk: int = 30,
        initial_buffer_size: int = 2,  # NEW: Configurable - frames before first yield (~160ms audio)
        buffer_size: int = 3,          # NEW: Configurable - frames per subsequent yield (~240ms audio)
        on_chunk_generated: Optional[Callable[[torch.Tensor], None]] = None,
    ):
        """
        Generate audio in a streaming fashion, optimized for lower latency to first chunk.
        
        Args:
            text: Text to generate audio for
            speaker: Speaker ID
            context: List of context segments for voice cloning
            max_audio_length_ms: Maximum audio length in milliseconds
            temperature: Sampling temperature (lower = more deterministic)
            topk: Top-k sampling parameter
            initial_buffer_size: Number of frames before first yield (default: 2, ~160ms audio)
            buffer_size: Number of frames per subsequent yield (default: 3, ~240ms audio)
            on_chunk_generated: Optional callback for each generated chunk
        
        Yields:
            torch.Tensor: Audio chunks as they are generated
        """
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)

        tokens, tokens_mask = [], []

        # Use the configurable buffer sizes
        current_buffer_size = initial_buffer_size
        first_chunk_delivered = False

        context_start = time.time()
        if context:
            for segment in context:
                segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
                tokens.append(segment_tokens)
                tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        max_seq_len = 2048
        if prompt_tokens.size(0) > max_seq_len:
            prompt_tokens = prompt_tokens[-max_seq_len:]
            prompt_tokens_mask = prompt_tokens_mask[-max_seq_len:]

        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        zeros_1_1 = torch.zeros(1, 1).long().to(self.device)
        zeros_mask_1_1 = torch.zeros(1, 1).bool().to(self.device)

        def update_tokens(sample):
            nonlocal curr_tokens, curr_tokens_mask, curr_pos
            ones = torch.ones_like(sample).bool()
            curr_tokens = torch.cat([sample, zeros_1_1], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat([ones, zeros_mask_1_1], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        generation_start = time.time()
        frames_generated = 0

        with self._audio_tokenizer.streaming(1):
            audio_buffer = []  # Accumulate decoded audio samples

            for i in range(max_generation_len):
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)

                if torch.all(sample == 0):
                    break

                frames_generated += 1
                update_tokens(sample)

                # Decode single frame immediately to avoid streaming state issues
                audio_frame = self._audio_tokenizer.decode(
                    sample.transpose(0, 1).unsqueeze(2)
                )
                audio_buffer.append(audio_frame.squeeze())

                # Yield when buffer is full
                if len(audio_buffer) >= current_buffer_size:
                    # Concatenate accumulated audio samples
                    audio_chunk = torch.cat(audio_buffer, dim=-1)
                    cpu_chunk = audio_chunk.cpu()

                    if on_chunk_generated:
                        on_chunk_generated(cpu_chunk)

                    # After first chunk, switch to normal buffer size
                    if not first_chunk_delivered:
                        current_buffer_size = buffer_size
                        first_chunk_delivered = True

                    yield cpu_chunk

                    # Clear buffer for next batch
                    audio_buffer = []

            # Yield any remaining audio
            if audio_buffer:
                audio_chunk = torch.cat(audio_buffer, dim=-1)
                cpu_chunk = audio_chunk.cpu()
                if on_chunk_generated:
                    on_chunk_generated(cpu_chunk)
                yield cpu_chunk

        # Print final performance metrics
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = time.time() - generation_start
        audio_seconds = frames_generated * 0.08
        rtf = total_time / audio_seconds if audio_seconds > 0 else float('inf')
        print(f"Total time: {total_time:.2f}s")
        print(f"Generated {frames_generated} frames ({audio_seconds:.2f}s of audio)")
        print(f"Real-time factor: {rtf:.3f}x (target: <1.0)")

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.8,
        topk: int = 40,
        stream: bool = False,
        output_file: Optional[str] = None,
    ):
        """
        Generate audio with optional streaming and file output.
        
        Args:
            text: Text to generate audio for
            speaker: Speaker ID
            context: List of context segments
            max_audio_length_ms: Maximum audio length in milliseconds
            temperature: Sampling temperature
            topk: Top-k sampling parameter
            stream: Whether to use streaming generation
            output_file: If provided and stream=True, output will be saved to this file
        
        Returns:
            torch.Tensor: Generated audio tensor
        """
        if stream:
            if output_file:
                # Setup streaming to file
                write_chunk, close_wav = stream_audio_to_wav(output_file, self.sample_rate)
                
                # Collect chunks while streaming to file
                audio_chunks = []
                t1 = time.time()
                
                for i, chunk in enumerate(self.generate_stream(
                    text, speaker, context, max_audio_length_ms, temperature, topk
                )):
                    # Write to file
                    write_chunk(chunk)
                    # Store for return value
                    audio_chunks.append(chunk)
                    
                    # Occasionally print progress
                    if i % 5 == 0:
                        print(f"Part {i+1} available after {time.time() - t1:.4f}s")
                        t1 = time.time()
                
                # Close file
                close_wav()
                print(f"Streaming complete, WAV file saved to {output_file}")
            else:
                # Just collect chunks without file output
                audio_chunks = []
                for chunk in self.generate_stream(text, speaker, context, max_audio_length_ms, temperature, topk):
                    audio_chunks.append(chunk)
            
            if not audio_chunks:
                return torch.tensor([])
            return torch.cat(audio_chunks)

        # Non-streaming generation remains unchanged
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []

        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        max_seq_len = 2048
        if prompt_tokens.size(0) > max_seq_len:
            prompt_tokens = prompt_tokens[-max_seq_len:]
            prompt_tokens_mask = prompt_tokens_mask[-max_seq_len:]

        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        samples = []
        with self._audio_tokenizer.streaming(1):
            for _ in range(max_generation_len):
                sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                if torch.all(sample == 0):
                    break
                samples.append(sample)

                curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
                curr_tokens_mask = torch.cat(
                    [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
                ).unsqueeze(1)
                curr_pos = curr_pos[:, -1:] + 1

        if not samples:
            return torch.tensor([])

        return self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)


def stream_audio_to_wav(filename, sample_rate):
    """
    Initialize a WAV writer for streaming audio chunks.
    """
    wav_file = wave.open(filename, 'wb')
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)
    
    def write_chunk(audio_chunk):
        if isinstance(audio_chunk, torch.Tensor):
            audio_np = audio_chunk.detach().cpu().numpy()
        else:
            audio_np = audio_chunk
            
        if audio_np.max() <= 1.0 and audio_np.min() >= -1.0:
            audio_int = (audio_np * 32767).astype(np.int16)
        else:
            audio_int = audio_np.astype(np.int16)
            
        wav_file.writeframes(audio_int.tobytes())
    
    def close():
        wav_file.close()
        
    return write_chunk, close
