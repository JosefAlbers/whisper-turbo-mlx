import base64
import json
import math
import os
from functools import lru_cache
from subprocess import CalledProcessError, run

import fire
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import tiktoken
from huggingface_hub import snapshot_download, hf_hub_download
import librosa

N_FFT = 400

class Tokenizer:
    def __init__(self):
        path_tok = 'multilingual.tiktoken'
        if not os.path.exists(path_tok):
            path_tok = hf_hub_download(repo_id='JosefAlbers/whisper', filename=path_tok)
        with open(path_tok) as f:
            ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in f if line)}
        n_vocab = len(ranks)
        specials = ["<|endoftext|>", "<|startoftranscript|>", *[f"<|_{lang}|>" for lang in range(100)], "<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>", *[f"<|{i * 0.02:.2f}|>" for i in range(1501)]]
        special_tokens = {k:(n_vocab+i) for i,k in enumerate(specials)}
        self.encoding = tiktoken.Encoding(name='jj', explicit_n_vocab=n_vocab + len(special_tokens), pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", mergeable_ranks=ranks, special_tokens=special_tokens)
    def encode(self, lot):
        if isinstance(lot, str):
            lot = [lot]
        return [self.encoding.encode(t, allowed_special='all') for t in lot]
    def decode(self, lol):
        if isinstance(lol[0], int):
            lol = [lol]
        return [self.encoding.decode(l) for l in lol]

def load_audio(file, sr=16000):
    try:
        out = run(["ffmpeg", "-nostdin", "-threads", "0", "-i", file, "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr), "-"], capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return mx.array(np.frombuffer(out, np.int16)).flatten().astype(mx.float32) / 32768.0

@lru_cache(maxsize=None)
def mel_filters(n_mels):
    path_mel = "mel_filters.npz"
    if not os.path.exists(path_mel):
        np.savez_compressed(path_mel, mel_128=librosa.filters.mel(sr=16000, n_fft=N_FFT, n_mels=128))
    return mx.load(path_mel)[f"mel_{n_mels}"]

@lru_cache(maxsize=None)
def hanning():
    return mx.array(np.hanning(N_FFT + 1)[:-1])

def stft(x, window, nperseg=400, noverlap=160, nfft=None, axis=-1, pad_mode="reflect"):
    if nfft is None:
        nfft = nperseg
    if noverlap is None:
        noverlap = nfft // 4
    def _pad(x, padding, pad_mode="constant"):
        if pad_mode == "constant":
            return mx.pad(x, [(padding, padding)])
        elif pad_mode == "reflect":
            prefix = x[1 : padding + 1][::-1]
            suffix = x[-(padding + 1) : -1][::-1]
            return mx.concatenate([prefix, x, suffix])
        else:
            raise ValueError(f"Invalid pad_mode {pad_mode}")
    padding = nperseg // 2
    x = _pad(x, padding, pad_mode)
    strides = [noverlap, 1]
    t = (x.size - nperseg + noverlap) // noverlap
    shape = [t, nfft]
    x = mx.as_strided(x, shape=shape, strides=strides)
    return mx.fft.rfft(x * window)

def log_mel_spectrogram(audio, n_mels=128, padding=480000):
    if isinstance(audio, str):
        audio = load_audio(audio)
    elif not isinstance(audio, mx.array):
        audio = mx.array(audio)
    if padding > 0:
        audio = mx.pad(audio, (0, padding))
    window = hanning()
    freqs = stft(audio, window, nperseg=N_FFT, noverlap=160)
    magnitudes = freqs[:-1, :].abs().square()
    filters = mel_filters(n_mels)
    mel_spec = magnitudes @ filters.T
    log_spec = mx.maximum(mel_spec, 1e-10).log10()
    log_spec = mx.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

def sinusoids(length, channels, max_timescale=10000):
    assert channels % 2 == 0
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = mx.exp(-log_timescale_increment * mx.arange(channels // 2))
    scaled_time = mx.arange(length)[:, None] * inv_timescales[None, :]
    return mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    def __call__(self, x, xa=None, mask=None, kv_cache=None):
        q = self.q_proj(x)
        if xa is None:
            k = self.k_proj(x)
            v = self.v_proj(x)
            if kv_cache is not None:
                k = mx.concatenate([kv_cache[0], k], axis=1)
                v = mx.concatenate([kv_cache[1], v], axis=1)
        elif kv_cache is None:
            k = self.k_proj(xa)
            v = self.v_proj(xa)
        else:
            k, v = kv_cache
        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out_proj(wv), (k, v), qk

    def qkv_attention(self, q, k, v, mask=None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.reshape(*q.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3) * scale
        k = k.reshape(*k.shape[:2], self.n_head, -1).transpose(0, 2, 3, 1) * scale
        v = v.reshape(*v.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3)
        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.astype(mx.float32)
        w = mx.softmax(qk, axis=-1).astype(q.dtype)
        out = (w @ v).transpose(0, 2, 1, 3)
        out = out.reshape(n_batch, n_ctx, n_state)
        return out, qk

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, cross_attention=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_attn = MultiHeadAttention(d_model, n_head) if cross_attention else None
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model) if cross_attention else None
        n_mlp = d_model * 4
        self.fc1 = nn.Linear(d_model, n_mlp)
        self.fc2 = nn.Linear(n_mlp, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)
    def __call__(self, x, xa=None, mask=None, kv_cache=None):
        kv, cross_kv = kv_cache if kv_cache else (None, None)
        y, kv, _ = self.self_attn(self.self_attn_layer_norm(x), mask=mask, kv_cache=kv)
        x += y
        cross_qk = None
        if self.encoder_attn:
            y, cross_kv, cross_qk = self.encoder_attn(self.encoder_attn_layer_norm(x), xa, kv_cache=cross_kv)
            x += y
        x = x + self.fc2(nn.gelu(self.fc1(self.final_layer_norm(x))))
        return x, (kv, cross_kv), cross_qk

class AudioEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv1 = nn.Conv1d(cfg['num_mel_bins'], cfg['d_model'], kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cfg['d_model'], cfg['d_model'], kernel_size=3, stride=2, padding=1)
        self._positional_embedding = sinusoids(cfg['max_source_positions'], cfg['d_model']).astype(mx.float16)
        self.layers = [ResidualAttentionBlock(cfg['d_model'], cfg['encoder_attention_heads']) for _ in range(cfg['encoder_layers'])]
        self.layer_norm = nn.LayerNorm(cfg['d_model'])
    def __call__(self, x):
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))
        x = x + self._positional_embedding
        for block in self.layers:
            x, _, _ = block(x)
        x = self.layer_norm(x)
        return x

class TextDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg['vocab_size'], cfg['d_model'])
        self.positional_embedding = mx.zeros((cfg['max_target_positions'], cfg['d_model']))
        self.layers = [ResidualAttentionBlock(cfg['d_model'], cfg['decoder_attention_heads'], cross_attention=True) for _ in range(cfg['decoder_layers'])]
        self.layer_norm = nn.LayerNorm(cfg['d_model'])
        self._mask = nn.MultiHeadAttention.create_additive_causal_mask(cfg['max_target_positions']).astype(mx.float16)
    def __call__(self, x, xa, kv_cache=None):
        offset = kv_cache[0][0][0].shape[1] if kv_cache else 0
        x = self.embed_tokens(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        if kv_cache is None:
            kv_cache = [None] * len(self.layers)
        cross_qk = [None] * len(self.layers)
        for e, block in enumerate(self.layers):
            x, kv_cache[e], cross_qk[e] = block(x, xa, mask=self._mask, kv_cache=kv_cache[e])
        x = self.layer_norm(x)
        return self.embed_tokens.as_linear(x), kv_cache, cross_qk

class Whisper(nn.Module):
    def __init__(self, cfg):
        self.encoder = AudioEncoder(cfg)
        self.decoder = TextDecoder(cfg)
    def __call__(self, mel, txt):
        return self.decoder(txt, self.encoder(mel))[0]
    def encode(self, mel):
        return self.encoder(mel)
    def decode(self, txt, mel, kv_cache):
        return self.decoder(txt, mel, kv_cache)

class Transcriber(nn.Module):
    def __init__(self, cfg):
        self.model = Whisper(cfg)
        self.tokenizer = Tokenizer()
    def __call__(self, path_audio, any_lang, quick):
        raw = log_mel_spectrogram(path_audio)
        sot = mx.array([[50258, 50360, 50365]]) if any_lang else mx.array([[50258, 50259, 50360, 50365]])
        txt = self.parallel(raw, sot) if quick else self.recurrent(raw, sot)
        return txt
    def recurrent(self, raw, sot):
        new_tok, i = mx.zeros((1,0), dtype=mx.int32), 0
        while i+3000 < len(raw):
            piece = self.step(raw[i:i+3000][None], sot)
            arg_hop = mx.argmax(piece).item()
            hop = (piece[:,arg_hop].astype(mx.int32).item()-50365)*2
            new_tok = mx.concatenate([new_tok, piece[:,:arg_hop]], axis=-1)
            i += hop if hop > 0 else 3000
        new_tok = [i for i in new_tok.astype(mx.int32).tolist()[0] if i < 50257]
        return self.tokenizer.decode(new_tok)[0]
    def parallel(self, raw, sot):
        raw = raw[:(raw.shape[0]//3000)*3000].reshape(-1, 3000, 128)
        sot = mx.repeat(sot, raw.shape[0], 0)
        new_tok = self.step(raw, sot)
        arg_hop = mx.argmax(new_tok, axis=-1).tolist()
        new_tok = [i[:a] for i,a in zip(new_tok.astype(mx.int32).tolist(),arg_hop)]
        new_tok = [i for i in sum(new_tok, []) if i < 50257]
        return self.tokenizer.decode(new_tok)[0]
    def step(self, mel, txt):
        mel = self.model.encode(mel)
        kv_cache = None
        B = mel.shape[0]
        new_tok = mx.zeros((B,0), dtype=mx.int32)
        goon = mx.ones((B,1), dtype=mx.bool_)
        for i in range(445):
            logits, kv_cache, _ = self.model.decode(txt=txt, mel=mel, kv_cache=kv_cache)
            txt = mx.argmax(logits[:,-1,:], axis=-1, keepdims=True)
            mx.eval(txt)
            new_tok = mx.concatenate([new_tok, txt], axis=-1)
            goon *= (txt != 50257)
            if goon.sum() <= 0:
                break
        return new_tok

def transcribe(path_audio=None, any_lang=False, quick=False):
    if path_audio is None:
        path_audio = hf_hub_download(repo_id='JosefAlbers/exurb1a', filename='1_alive.mp3')
    path_hf = snapshot_download(repo_id='openai/whisper-large-v3-turbo', allow_patterns=["config.json", "model.safetensors"])
    with open(f'{path_hf}/config.json', 'r') as fp:
        cfg = json.load(fp)
    weights = [(k.replace("embed_positions.weight", "positional_embedding"), v.swapaxes(1, 2) if ('conv' in k and v.ndim==3) else v) for k, v in mx.load(f'{path_hf}/model.safetensors').items()]
    model = Transcriber(cfg)
    model.load_weights(weights, strict=False)
    model.eval()
    mx.eval(model)
    return model(path_audio=path_audio, any_lang=any_lang, quick=quick)

def fire_main():
    fire.Fire(transcribe)

if __name__ == '__main__':
    fire.Fire(transcribe)

    # for path_audio in ['0_test.wav', None]:
    #     for any_lang in [True, False]:
    #         for quick in [True, False]:
    #             print(transcribe(path_audio=path_audio, any_lang=any_lang, quick=quick))
