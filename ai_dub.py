#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from faster_whisper import WhisperModel
from piper.voice import PiperVoice
from piper import config as pconfig

FFMPEG_BIN = str(Path('/workspace/tools/ffmpeg/ffmpeg'))
FFPROBE_BIN = str(Path('/workspace/tools/ffmpeg/ffprobe'))


@dataclass
class Segment:
    start: float
    end: float
    text: str


def run_cmd(cmd: List[str], *, input_bytes: Optional[bytes] = None) -> bytes:
    proc = subprocess.run(
        cmd,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(shlex.quote(c) for c in cmd)}\nSTDERR:\n{proc.stderr.decode(errors='ignore')[:4000]}"
        )
    return proc.stdout


def ffprobe_duration(path: str) -> float:
    cmd = [
        FFPROBE_BIN,
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        path,
    ]
    out = run_cmd(cmd).decode().strip()
    try:
        return float(out)
    except Exception:
        return 0.0


def extract_audio_pcm16le(path: str, sample_rate: int = 16000) -> np.ndarray:
    cmd = [
        FFMPEG_BIN,
        '-y',
        '-i', path,
        '-vn',
        '-ac', '1',
        '-ar', str(sample_rate),
        '-f', 's16le',
        'pipe:1',
    ]
    raw = run_cmd(cmd)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def numpy_float_to_s16le_bytes(audio: np.ndarray) -> bytes:
    a = np.clip(audio, -1.0, 1.0)
    a = (a * 32767.0).astype(np.int16)
    return a.tobytes()


def ffmpeg_tempo_adjust(
    audio: np.ndarray,
    sample_rate: int,
    tempo_ratio: float,
) -> np.ndarray:
    if len(audio) == 0:
        return audio
    # Build atempo chain in [0.5,2.0]
    filters = []
    r = float(tempo_ratio)
    # Avoid pathological cases
    r = max(0.1, min(10.0, r))
    while r > 2.0:
        filters.append('atempo=2.0')
        r /= 2.0
    while r < 0.5:
        filters.append('atempo=0.5')
        r /= 0.5
    filters.append(f'atempo={r:.6f}')
    afilter = ','.join(filters)

    in_bytes = numpy_float_to_s16le_bytes(audio)
    cmd = [
        FFMPEG_BIN,
        '-y',
        '-f', 's16le',
        '-ar', str(sample_rate),
        '-ac', '1',
        '-i', 'pipe:0',
        '-filter:a', afilter,
        '-f', 's16le',
        'pipe:1',
    ]
    out = run_cmd(cmd, input_bytes=in_bytes)
    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0


def ffmpeg_resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return audio
    in_bytes = numpy_float_to_s16le_bytes(audio)
    cmd = [
        FFMPEG_BIN,
        '-y',
        '-f', 's16le',
        '-ar', str(from_sr),
        '-ac', '1',
        '-i', 'pipe:0',
        '-ar', str(to_sr),
        '-ac', '1',
        '-f', 's16le',
        'pipe:1',
    ]
    out = run_cmd(cmd, input_bytes=in_bytes)
    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0


def write_wav_via_ffmpeg(audio: np.ndarray, sample_rate: int, out_path: str) -> None:
    in_bytes = numpy_float_to_s16le_bytes(audio)
    cmd = [
        FFMPEG_BIN,
        '-y',
        '-f', 's16le',
        '-ar', str(sample_rate),
        '-ac', '1',
        '-i', 'pipe:0',
        '-ar', str(sample_rate),
        '-ac', '1',
        out_path,
    ]
    _ = run_cmd(cmd, input_bytes=in_bytes)


def mux_audio_to_video(video_path: str, audio_path: str, out_path: str) -> None:
    # Keep original video, replace audio.
    # Use libopus for .webm and .mkv containers; use aac for .mp4 family.
    ext = Path(out_path).suffix.lower()
    if ext in ('.webm', '.mkv'):
        audio_codec = 'libopus'
    else:
        audio_codec = 'aac'
    cmd = [
        FFMPEG_BIN,
        '-y',
        '-i', video_path,
        '-i', audio_path,
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-c:v', 'copy',
        '-c:a', audio_codec,
        '-shortest',
        out_path,
    ]
    run_cmd(cmd)


def load_piper_voice(model_path: str, config_path: Optional[str]) -> PiperVoice:
    return PiperVoice.load(model_path=model_path, config_path=config_path, use_cuda=False)


def piper_synthesize(voice: PiperVoice, text: str, syn_cfg: Optional[pconfig.SynthesisConfig]) -> Tuple[np.ndarray, int]:
    # Accumulate float audio from chunks
    samples: List[np.ndarray] = []
    sample_rate: Optional[int] = None
    for chunk in voice.synthesize(text, syn_cfg):
        if sample_rate is None:
            sample_rate = int(chunk.sample_rate)
        if chunk.audio_float_array is not None and len(chunk.audio_float_array) > 0:
            samples.append(chunk.audio_float_array.astype(np.float32))
        elif chunk._audio_int16_array is not None:
            samples.append((chunk._audio_int16_array.astype(np.float32) / 32768.0))
    if sample_rate is None:
        # Fallback to common Piper rate if unknown
        sample_rate = 22050
    if not samples:
        return np.zeros(0, dtype=np.float32), sample_rate
    audio = np.concatenate(samples, axis=0)
    return audio, sample_rate


def transcribe_segments(model: WhisperModel, audio: np.ndarray, sr: int, language: Optional[str]) -> List[Segment]:
    segments: List[Segment] = []
    it, info = model.transcribe(audio, language=language, word_timestamps=False, vad_filter=True)
    for seg in it:
        text = seg.text.strip()
        if text:
            segments.append(Segment(start=seg.start, end=seg.end, text=text))
    return segments


def sanitize_text(text: str) -> str:
    # Remove Whisper bracket annotations and trim excessive spaces.
    t = re.sub(r"\s+", " ", text)
    t = re.sub(r"\[(music|noise|applause|laughter|silence)\]", " ", t, flags=re.I)
    return t.strip()


def build_aligned_audio(
    segments: List[Segment],
    video_duration: float,
    voice: PiperVoice,
    syn_cfg: Optional[pconfig.SynthesisConfig],
    out_sr: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    timeline: List[np.ndarray] = []
    current_time = 0.0
    target_sr: Optional[int] = out_sr

    for seg in segments:
        start, end = float(seg.start), float(seg.end)
        if end <= start:
            continue
        # Prepend silence if gap
        if start > current_time:
            gap = start - current_time
            if target_sr is None:
                # Lazy init to Piper's sr later; for now skip adding silence, we will add once SR known
                pass
            else:
                pad_len = int(round(gap * target_sr))
                if pad_len > 0:
                    timeline.append(np.zeros(pad_len, dtype=np.float32))
            current_time = start

        text = sanitize_text(seg.text)
        if not text:
            # Pure silence segment
            if target_sr is not None:
                pad_len = int(round((end - start) * target_sr))
                if pad_len > 0:
                    timeline.append(np.zeros(pad_len, dtype=np.float32))
            current_time = end
            continue

        tts_audio, tts_sr = piper_synthesize(voice, text, syn_cfg)
        if target_sr is None:
            target_sr = int(tts_sr)
            # Now that SR is known, we must insert initial silence up to start
            if start > 0:
                pad_len0 = int(round(start * target_sr))
                if pad_len0 > 0:
                    timeline.append(np.zeros(pad_len0, dtype=np.float32))
        if tts_sr != target_sr:
            tts_audio = ffmpeg_resample(tts_audio, from_sr=tts_sr, to_sr=target_sr)

        required_dur = end - start
        tts_dur = len(tts_audio) / float(target_sr) if len(tts_audio) > 0 else 0.0
        if tts_dur <= 0.0:
            pad_len = int(round(required_dur * target_sr))
            if pad_len > 0:
                timeline.append(np.zeros(pad_len, dtype=np.float32))
            current_time = end
            continue

        ratio = required_dur / tts_dur
        adj = ffmpeg_tempo_adjust(tts_audio, sample_rate=target_sr, tempo_ratio=ratio)
        # Trim/pad to exact duration
        exact_len = int(round(required_dur * target_sr))
        if len(adj) > exact_len:
            adj = adj[:exact_len]
        elif len(adj) < exact_len:
            pad = np.zeros(exact_len - len(adj), dtype=np.float32)
            adj = np.concatenate([adj, pad], axis=0)
        timeline.append(adj)
        current_time = end

    if target_sr is None:
        # No voiced segments; return silence
        target_sr = 22050
        total_len = int(round(video_duration * target_sr))
        return np.zeros(total_len, dtype=np.float32), target_sr

    # Tail pad up to video duration
    if video_duration > current_time:
        pad_len = int(round((video_duration - current_time) * target_sr))
        if pad_len > 0:
            timeline.append(np.zeros(pad_len, dtype=np.float32))

    if not timeline:
        return np.zeros(0, dtype=np.float32), target_sr

    return np.concatenate(timeline, axis=0), target_sr


def process_video(
    video_path: str,
    whisper_model: WhisperModel,
    piper_voice: PiperVoice,
    syn_cfg: Optional[pconfig.SynthesisConfig],
    out_dir: Path,
    tmp_dir: Path,
    language: Optional[str],
    model_sr: int = 16000,
) -> Path:
    print(f"[i] Processing: {video_path}")
    duration = ffprobe_duration(video_path)
    print(f"[i] Duration: {duration:.2f}s")

    audio = extract_audio_pcm16le(video_path, sample_rate=model_sr)
    print(f"[i] Audio samples: {len(audio)} @ {model_sr}Hz")

    segs = transcribe_segments(whisper_model, audio, model_sr, language)
    print(f"[i] Segments: {len(segs)}")

    aligned_audio, aligned_sr = build_aligned_audio(
        segs, video_duration=duration, voice=piper_voice, syn_cfg=syn_cfg
    )
    print(f"[i] Aligned audio: {len(aligned_audio)} samples @ {aligned_sr}Hz")

    # Normalize to avoid clipping
    if len(aligned_audio) > 0:
        peak = float(np.max(np.abs(aligned_audio)))
        if peak > 0.99:
            aligned_audio = aligned_audio / (peak + 1e-6) * 0.95

    # Write temp wav
    tmp_wav = tmp_dir / (Path(video_path).stem + '.ai.wav')
    write_wav_via_ffmpeg(aligned_audio, aligned_sr, str(tmp_wav))

    # Mux back into video
    out_ext = Path(video_path).suffix
    # If input is .webm but contains H.264 video (common for some screen recorders),
    # writing .webm is invalid. Default to Matroska container.
    out_ext2 = '.mkv' if out_ext.lower() == '.webm' else out_ext
    out_path = out_dir / (Path(video_path).stem + '_ai' + out_ext2)
    mux_audio_to_video(video_path, str(tmp_wav), str(out_path))

    print(f"[✓] Wrote: {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description='Convert video audio to AI TTS while preserving timing')
    ap.add_argument('--videos', nargs='*', help='Video files to process (.webm, .mp4). Default: all in CWD')
    ap.add_argument('--voice', default='voices/en_US-amy-medium.onnx', help='Path to Piper ONNX voice model')
    ap.add_argument('--voice-config', default='voices/en_US-amy-medium.onnx.json', help='Path to Piper voice JSON config')
    ap.add_argument('--whisper-model', default='tiny', help='faster-whisper model size or path (e.g., tiny, base, small)')
    ap.add_argument('--language', default=None, help='Language code (auto if omitted)')
    ap.add_argument('--out-dir', default='ai_outputs', help='Directory for outputs')
    ap.add_argument('--tmp-dir', default='.ai_tmp', help='Directory for temp files')
    ap.add_argument('--length-scale', type=float, default=1.0, help='Piper length scale (baseline speaking rate)')
    ap.add_argument('--noise-scale', type=float, default=0.667, help='Piper noise scale')
    ap.add_argument('--noise-w-scale', type=float, default=0.8, help='Piper noise w scale')
    ap.add_argument('--volume', type=float, default=1.0, help='Piper gain multiplier')
    args = ap.parse_args()

    # Resolve videos
    if not args.videos:
        vids = [
            *sorted(Path('.').glob('*.webm')),
            *sorted(Path('.').glob('*.mp4')),
            *sorted(Path('.').glob('*.mkv')),
        ]
    else:
        vids = [Path(v) for v in args.videos]
    vids = [str(v) for v in vids if v.exists()]
    if not vids:
        print('No videos found to process.', file=sys.stderr)
        sys.exit(1)

    # Check tools
    if not Path(FFMPEG_BIN).exists():
        print(f'ffmpeg binary not found at {FFMPEG_BIN}', file=sys.stderr)
        sys.exit(2)
    if not Path(FFPROBE_BIN).exists():
        print(f'ffprobe binary not found at {FFPROBE_BIN}', file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print('[i] Loading Whisper model:', args.whisper_model)
    whisper = WhisperModel(args.whisper_model, device='cpu', compute_type='int8')

    print('[i] Loading Piper voice:', args.voice)
    if not Path(args.voice).exists():
        print(f'Voice model not found: {args.voice}', file=sys.stderr)
        sys.exit(3)
    voice_cfg_path = args.voice_config if args.voice_config and Path(args.voice_config).exists() else None
    voice = load_piper_voice(args.voice, voice_cfg_path)

    syn_cfg = pconfig.SynthesisConfig(
        speaker_id=None,
        length_scale=args.length_scale,
        noise_scale=args.noise_scale,
        noise_w_scale=args.noise_w_scale,
        normalize_audio=True,
        volume=args.volume,
    )

    generated: List[Path] = []
    for vpath in vids:
        try:
            outp = process_video(
                video_path=vpath,
                whisper_model=whisper,
                piper_voice=voice,
                syn_cfg=syn_cfg,
                out_dir=out_dir,
                tmp_dir=tmp_dir,
                language=args.language,
            )
            generated.append(outp)
        except Exception as e:
            print(f"[!] Failed for {vpath}: {e}", file=sys.stderr)

    print('\n[i] Done. Generated:')
    for p in generated:
        print(' -', p)


if __name__ == '__main__':
    main()
