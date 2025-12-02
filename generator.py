"""
Optimized generate_stream method for real-time audio streaming.
Target: ~480-500ms first chunk latency with real-time playback.

Key changes from original:
1. Reduced buffer sizes (6-8 instead of 20)
2. Removed unnecessary torch.cuda.synchronize() calls
3. Optimized for lower latency over throughput
"""

# === RECOMMENDED PARAMETER VALUES ===

# In generate_stream method:
INITIAL_BATCH_SIZE = 6      # Frames to generate before first buffer check
NORMAL_BATCH_SIZE = 8       # Frames per batch after first chunk
INITIAL_BUFFER_SIZE = 6     # Frames before first decode (~480ms)
NORMAL_BUFFER_SIZE = 8      # Frames per chunk after first (~640ms)

# In load_csm_1b_local:
AUDIO_NUM_CODEBOOKS = 32    # Reduced from 32 (quality vs speed tradeoff)

# In generation calls:
RECOMMENDED_TOPK = 20       # Reduced from 30-50
RECOMMENDED_TEMPERATURE = 0.7

# Generator setting:
STREAM_BUFFER_SIZE = 6      # Reduced from 20


# === MODIFIED generate_stream METHOD ===
# Replace the generate_stream method in your Generator class with this:

"""
@torch.inference_mode()
def generate_stream(
    self,
    text: str,
    speaker: int,
    context: List[Segment],
    max_audio_length_ms: float = 90_000,
    temperature: float = 0.7,
    topk: int = 20,  # Reduced default
    on_chunk_generated: Optional[Callable[[torch.Tensor], None]] = None,
):
    '''
    Generate audio in a streaming fashion, optimized for real-time playback.
    Target: ~480ms first chunk latency.
    '''
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        # REMOVED: torch.cuda.synchronize() - unnecessary here

    self._model.reset_caches()

    max_generation_len = int(max_audio_length_ms / 80)

    tokens, tokens_mask = [], []

    # === OPTIMIZED PARAMETERS ===
    initial_batch_size = 6      # was 20
    normal_batch_size = 8       # was 20
    initial_buffer_size = 6     # was 20 → ~480ms first chunk
    normal_buffer_size = 8      # was 20
    
    batch_size = initial_batch_size
    buffer_size = initial_buffer_size
    first_chunk_delivered = False

    # Context processing (unchanged)
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

    expected_frame_count = buffer_size 
    frame_buffer = []

    # Pre-allocate tensors (optimization)
    zeros_1_1 = torch.zeros(1, 1, dtype=torch.long, device=self.device)
    zeros_mask_1_1 = torch.zeros(1, 1, dtype=torch.bool, device=self.device)

    def update_tokens(sample):
        nonlocal curr_tokens, curr_tokens_mask, curr_pos
        ones = torch.ones_like(sample, dtype=torch.bool)
        curr_tokens = torch.cat([sample, zeros_1_1], dim=1).unsqueeze(1)
        curr_tokens_mask = torch.cat([ones, zeros_mask_1_1], dim=1).unsqueeze(1)
        curr_pos = curr_pos[:, -1:] + 1

    with self._audio_tokenizer.streaming(1):
        i = 0
        generation_start = time.time()

        while i < max_generation_len:
            batch_end = min(i + batch_size, max_generation_len)
            batch_size_actual = batch_end - i

            batch_samples = []

            for _ in range(batch_size_actual):
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                    
                    # REMOVED: torch.cuda.synchronize() and complex checks
                    # Simple check is enough:
                    if sample.numel() == 0:
                        break
                        
                if torch.all(sample == 0):
                    break

                batch_samples.append(sample)
                update_tokens(sample)

            if not batch_samples:
                break

            frame_buffer.extend(batch_samples)
            i += len(batch_samples)

            # Check if we have enough frames to decode
            if len(frame_buffer) >= buffer_size:
                frames_to_process = frame_buffer[:expected_frame_count]
                
                # Pad if needed
                if len(frames_to_process) < expected_frame_count:
                    padding_frames = [
                        torch.zeros_like(frames_to_process[0]) 
                        for _ in range(expected_frame_count - len(frames_to_process))
                    ]
                    frames_to_process = frames_to_process + padding_frames
                
                # Decode audio chunk
                frames_stacked = torch.stack(frames_to_process).permute(1, 2, 0)
                audio_chunk = self._audio_tokenizer.decode(frames_stacked).squeeze(0).squeeze(0)
                
                frame_buffer = frame_buffer[expected_frame_count:]
                
                cpu_chunk = audio_chunk.cpu()
                if on_chunk_generated:
                    on_chunk_generated(cpu_chunk)
                
                # Switch to normal sizes after first chunk
                if not first_chunk_delivered:
                    batch_size = normal_batch_size
                    buffer_size = normal_buffer_size
                    expected_frame_count = buffer_size
                    first_chunk_delivered = True
                
                yield cpu_chunk

                # REDUCED: Only print every 200 frames instead of 100
                if i >= 200 and (i % 200 == 0):
                    print(f"Generated {i} frames ({i * 0.08:.2f}s of audio)")

        # Process remaining frames (unchanged logic, just cleaner)
        if frame_buffer:
            if len(frame_buffer) < expected_frame_count:
                padding_frames = [
                    torch.zeros_like(frame_buffer[0]) 
                    for _ in range(expected_frame_count - len(frame_buffer))
                ]
                frames_to_process = frame_buffer + padding_frames
                actual_frames = len(frame_buffer)
            else:
                frames_multiple = (len(frame_buffer) // expected_frame_count) * expected_frame_count
                frames_to_process = frame_buffer[:frames_multiple]
                actual_frames = frames_multiple
                
            frames_stacked = torch.stack(frames_to_process).permute(1, 2, 0)
            audio_chunk = self._audio_tokenizer.decode(frames_stacked).squeeze(0).squeeze(0)
            
            # Trim padding if we added any
            if len(frame_buffer) < expected_frame_count:
                actual_samples = int(audio_chunk.shape[0] * (actual_frames / expected_frame_count))
                audio_chunk = audio_chunk[:actual_samples]
                
            cpu_chunk = audio_chunk.cpu()
            if on_chunk_generated:
                on_chunk_generated(cpu_chunk)
            yield cpu_chunk

        # Final metrics
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Only sync at the very end
        total_time = time.time() - generation_start
        frames_generated = i
        audio_seconds = frames_generated * 0.08
        rtf = total_time / audio_seconds if audio_seconds > 0 else float('inf')
        print(f"Total: {total_time:.2f}s | Audio: {audio_seconds:.2f}s | RTF: {rtf:.3f}x")
"""


# === QUICK PATCH INSTRUCTIONS ===
"""
To apply these changes to your existing code:

1. In generate_stream(), change these lines (around line 144):
   
   OLD:
   initial_batch_size = 20
   normal_batch_size = 20  
   initial_buffer_size = 20
   normal_buffer_size = 20
   
   NEW:
   initial_batch_size = 6
   normal_batch_size = 8  
   initial_buffer_size = 6
   normal_buffer_size = 8

2. In load_csm_1b_local(), change audio_num_codebooks:
   
   OLD:
   audio_num_codebooks: int = 32
   
   NEW:
   audio_num_codebooks: int = 16

3. Comment out or remove torch.cuda.synchronize() calls inside the 
   generation loop (keep the final one).

4. When calling generate_stream or generate_streaming_audio, use:
   temperature=0.7, topk=20

5. After creating the generator, set:
   generator._stream_buffer_size = 6
"""


# === EXPECTED RESULTS ===
"""
With these settings on a decent GPU (RTX 3080/4080 class):

- First chunk latency: ~400-500ms (was ~1.6s+)
- RTF: ~0.7-0.9x (should be under 1.0 for real-time)
- Audio quality: Slightly reduced due to fewer codebooks

If RTF is still > 1.0:
- Try audio_num_codebooks = 8 (more quality loss)
- Try topk = 10 (faster sampling)
- Ensure torch.compile() is working (first run is slow)
"""
