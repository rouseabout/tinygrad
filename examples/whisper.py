#!/usr/bin/env python3
# pip install librosa tiktoken
# ffmpeg -i https://2001archive.files.wordpress.com/2015/07/cantdo.wav -ar 16000 test16.wav
# python3 examples/whisper.py
# (wav files must be 16kHz, mono, <= 30 seconds)
import base64
import librosa
import math
import numpy as np
from pathlib import Path
from tinygrad.helpers import dtypes
from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d, Embedding, LayerNorm, Linear
from extra.utils import download_file, fake_torch_load_zipped, get_child
import tiktoken
import wave

def load_audio(fn):
    w = wave.open(fn, "rb")
    assert w.getnchannels() == 1 and w.getframerate() == 16000
    raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
N_FRAMES = N_SAMPLES // HOP_LENGTH

# FIXME: use https://github.com/geohot/tinygrad/pull/865
def log_mel_spectrogram(samples):
    filters = Tensor(librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS))

    stft = librosa.stft(samples, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann", pad_mode="reflect")
    magnitudes = Tensor(np.abs(stft[..., :-1])) ** 2

    mels = filters @ magnitudes
    mels = mels.clip(1e-10, 1e10)
    mels = mels.log() / math.log(10)
    mels = mels.maximum(mels.max() - 8.0)
    mels = (mels + 4.0) / 4.0
    return mels

LANGUAGES=["en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"]

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|startoftranscript|>",
    *[f"<|{l}|>" for l in LANGUAGES],
    "<|translate|>",
    "<|transcribe|>",
    "<|startoflm|>",
    "<|startofprev|>",
    "<|nospeech|>",
    "<|notimestamps|>",
    *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
]

def build_tokenizer(multilingual):
    basename = "multilingual.tiktoken" if multilingual else "gpt2.tiktoken"
    fn = Path(__file__).parent.parent / "weights" / f"{basename}.pt"
    download_file(f"https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/{basename}", fn)
    with open(fn, "rb") as f:
        ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in f.read().splitlines() if line)}

    n_vocab = len(ranks)
    special_tokens = {}
    for token in SPECIAL_TOKENS:
        special_tokens[token] = n_vocab
        n_vocab += 1

    return tiktoken.Encoding(
        name="blah",
        explicit_n_vocab=n_vocab,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens)

class MultiHeadAttention:
    def __init__(self, n_state, n_head):
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def __call__(self, x, xa=None, mask=None):
        q = self.query(x)
        k = self.key(x if xa is None else xa)
        v = self.value(x if xa is None else xa)

        _, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.reshape((*q.shape[:2], self.n_head, -1)).permute(0, 2, 1, 3) * scale
        k = k.reshape((*k.shape[:2], self.n_head, -1)).permute(0, 2, 3, 1) * scale
        v = v.reshape((*v.shape[:2], self.n_head, -1)).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]

        w = qk.softmax(axis=-1)
        wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

        return self.out(wv)

def eval_sequence(blocks, x, **kwargs):
    for block in blocks:
        x = block(x, **kwargs)
    return x

class ResidualAttentionBlock:
    def __init__(self, n_state, n_head, cross_attention=False):
        self.attn_ln = LayerNorm(n_state)
        self.attn = MultiHeadAttention(n_state, n_head)
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None
        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.mlp_ln = LayerNorm(n_state)
        self.mlp = [Linear(n_state, n_state * 4), Tensor.gelu, Linear(n_state * 4, n_state)]

    def __call__(self, x, xa=None, mask=None):
        x = x + self.attn(self.attn_ln(x), mask=mask)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)[0]
        x = x + eval_sequence(self.mlp, self.mlp_ln(x))
        return x

class AudioEncoder:
    def __init__(self, n_mels, n_ctx, n_state, n_head, n_layer):
        self.conv1 = Conv2d(n_mels, n_state, kernel_size=(3,), padding=1)
        self.conv2 = Conv2d(n_state, n_state, kernel_size=(3,), stride=2, padding=1)
        self.positional_embedding = Tensor.empty(n_ctx, n_state)
        self.blocks = [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        self.ln_post = LayerNorm(n_state)

    def __call__(self, x): # x=(batch_size, n_mels, n_ctx)
        x = self.conv1(x).gelu()
        x = self.conv2(x).gelu()
        x = x.permute(0, 2, 1)
        x = x + self.positional_embedding
        x = eval_sequence(self.blocks, x)
        x = self.ln_post(x)
        return x

class TextDecoder:
    def __init__(self, n_vocab, n_ctx, n_state, n_head, n_layer):
        self.token_embedding = Embedding(n_vocab, n_state)
        self.positional_embedding = Tensor.empty(n_ctx, n_state)
        self.blocks = [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        self.mask = Tensor(np.triu(np.ones((n_ctx, n_ctx), dtype=np.float32) * -np.inf, k=1))
        self.ln = LayerNorm(n_state)

    def __call__(self, x, xa): # x=(batch_size, <=n_ctx), xa=(batch_size, n_mels, n_audio_ctx)
        # FIXME: cast needed here otherwise codegen falls over
        x = self.token_embedding(x.cast(dtypes.float32)) + self.positional_embedding[: x.shape[-1]]
        x = eval_sequence(self.blocks, x, xa=xa, mask=self.mask)
        x = self.ln(x)
        x = x @ self.token_embedding.weight.transpose(0, 1)
        return x

def all_nz(x): # FIXME: must be a better way to do this
    return x.min().numpy() > 0 and x.max().numpy() > 0

class Model:
    def __init__(self, dims):
        self.n_text_ctx = dims["n_text_ctx"]
        self.encoder = AudioEncoder(dims["n_mels"], dims["n_audio_ctx"], dims["n_audio_state"], dims["n_audio_head"], dims["n_audio_layer"])
        self.decoder = TextDecoder(dims["n_vocab"], dims["n_text_ctx"], dims["n_text_state"], dims["n_text_head"], dims["n_text_layer"])

    def run(self, tokenizer, prompt, mels): # mels=(batch_size, n_mels, n_ctx)
        eot = tokenizer.eot_token
        tokens = Tensor([prompt], dtype=dtypes.int32).repeat((mels.shape[0], 1))
        audio_features = self.encoder(mels)

        while True:
            logits = self.decoder(tokens, audio_features)
            next_token = np.argmax(logits[:, -1].numpy(), axis=-1).astype(np.int32) # be greedy

            # if the sentence already ends with eot, force next_token as eot
            for i, has_eot in enumerate((tokens[:, -1] == eot).numpy()):
                if has_eot:
                    next_token[i] = eot

            tokens = tokens.cat(Tensor(next_token[:, None]), dim=-1)

            tokens.realize() # FIXME: without this, all_nz doesn't work, why?
            if all_nz(tokens[:, -1] == eot) or tokens.shape[-1] > self.n_text_ctx:
                break

        # FIXME: replace("eot") is only neccessary because tinygrad can't yet do something like:
        #   batch.masked_select((batch != eot))

        return [tokenizer.decode(batch.numpy()).replace("<|endoftext|>", "").strip() for batch in tokens[:, len(prompt):]]

WEIGHTS_URL_HASH = {
    "tiny.en": "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03",
    "tiny": "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9",
    "base.en": "25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead",
}

def transcribe(model_name, audios):
    fn = Path(__file__).parent.parent / "weights" / f"{model_name}.pt"
    download_file(f"https://openaipublic.azureedge.net/main/whisper/models/{WEIGHTS_URL_HASH[model_name]}/{model_name}.pt", fn)

    weights = fake_torch_load_zipped(fn)

    model = Model(weights["dims"])

    for k, v in weights["model_state_dict"].items():
        mv = get_child(model, k)
        mv.assign(v)

    samples = np.empty([len(audios), N_SAMPLES], dtype=np.float32)
    for i in range(len(audios)):
        raw = load_audio(audios[i])
        assert raw.shape[0] <= N_SAMPLES, f"{audio_paths[i]} > {CHUNK_LENGTH} seconds"
        raw.resize(N_SAMPLES)
        samples[i] = raw

    mels = log_mel_spectrogram(samples)

    tokenizer = build_tokenizer(weights["dims"]["n_vocab"] == 51865)

    prompt = [tokenizer.encode_single_token(w) for w in ["<|startoftranscript|>", "<|notimestamps|>"]]

    return model.run(tokenizer, prompt, mels)

X=["test16.wav"]
Y = transcribe("tiny.en", X)
for x, y in zip(X, Y):
    print(f"{x}: {y}")
