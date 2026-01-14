import os
import time
import queue
import json
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque, List

import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
import requests

# ===== GPT API 設定 =====
GPT_API_BASE_URL = os.getenv("GPT_API_BASE_URL", "").rstrip("/")
GPT_API_KEY = os.getenv("GPT_API_KEY", "")
GPT_MODEL_ID = os.getenv("GPT_MODEL_ID", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "你是一個語音助理，請用繁體中文回答使用者的問題，"
    "回答要簡潔、口語、適合直接唸出來。"
)


# ========= 設定區 =========
SAMPLE_RATE = 16000          # WebRTC VAD 建議 8k/16k/32k/48k
CHANNELS = 1
FRAME_MS = 20                # WebRTC VAD 支援 10/20/30ms
VAD_MODE = 2                 # 0~3 越大越嚴格（誤觸發更少，但可能漏）
PRE_ROLL_MS = 300            # 開口前緩衝（避免切到第一個字）
SILENCE_END_MS = 900         # 靜音多久視為一句話結束
MIN_UTTERANCE_MS = 400       # 太短的片段不送（避免噪音誤觸發）
MAX_UTTERANCE_MS = 15000     # 最長一句話（避免無限錄）
OUTPUT_DIR = "recordings"    # 存 wav 的資料夾
DEVICE_INDEX: Optional[int] = None  # None=預設麥克風；或填整數 index


# Whisper API（用環境變數，也可直接改常數）
SPEECHES_BASE_URL = os.getenv("SPEACHES_BASE_URL", "").rstrip("/")
TRANSCRIPTION_MODEL_ID = os.getenv("TRANSCRIPTION_MODEL_ID", "whisper-small")

API_TIMEOUT_SEC = 60

def ensure_gpt_env():
    if not GPT_API_BASE_URL:
        raise RuntimeError("請設定 GPT_API_BASE_URL")
    if not GPT_MODEL_ID:
        raise RuntimeError("請設定 GPT_MODEL_ID")

# ========= 程式主體 =========
@dataclass
class Segment:
    wav_path: str
    started_at: float
    ended_at: float


def ensure_env():
    if not SPEECHES_BASE_URL:
        raise RuntimeError(
            "請設定環境變數 SPEECHES_BASE_URL，例如：\n"
            "  export SPEECHES_BASE_URL='http://127.0.0.1:8000'\n"
            "或在程式中直接指定 SPEECHES_BASE_URL 常數。"
        )


def pcm16_bytes_from_float32(x: np.ndarray) -> bytes:
    """
    sounddevice callback 給的通常是 float32 [-1,1]，轉成 PCM16 bytes 給 webrtcvad。
    x: shape (n, 1) 或 (n,)
    """
    if x.ndim == 2:
        x = x[:, 0]
    x = np.clip(x, -1.0, 1.0)
    pcm16 = (x * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def save_wav(path: str, audio_float32: np.ndarray, samplerate: int = SAMPLE_RATE):
    # audio_float32 shape (n,1) for soundfile
    sf.write(path, audio_float32, samplerate, subtype="PCM_16")


def call_whisper_api(wav_path: str) -> dict:
    """
    等效於：
    curl -s "$SPEACHES_BASE_URL/v1/audio/transcriptions" \
      -F "file=@audio.wav" -F "model=$TRANSCRIPTION_MODEL_ID"
    """
    url = f"{SPEECHES_BASE_URL}/v1/audio/transcriptions"
    with open(wav_path, "rb") as f:
        files = {"file": (os.path.basename(wav_path), f, "audio/wav")}
        data = {"model": TRANSCRIPTION_MODEL_ID}
        r = requests.post(url, files=files, data=data, timeout=API_TIMEOUT_SEC)
    r.raise_for_status()
    # 常見回傳：{"text":"..."} 或更完整 json
    try:
        return r.json()
    except Exception:
        return {"raw": r.text}

def call_gpt_api(user_text: str) -> str:
    """
    OpenAI Chat Completions 相容 API
    """
    url = f"{GPT_API_BASE_URL}/chat/completions"

    headers = {
        "Content-Type": "application/json",
    }
    if GPT_API_KEY:
        headers["Authorization"] = f"Bearer {GPT_API_KEY}"

    payload = {
        "model": GPT_MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.4,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()

    data = r.json()
    return data["choices"][0]["message"]["content"]


def main():
    ensure_env()
    ensure_gpt_env()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    vad = webrtcvad.Vad(VAD_MODE)

    frame_samples = int(SAMPLE_RATE * FRAME_MS / 1000)  # 20ms @16k => 320 samples
    silence_frames_end = int(SILENCE_END_MS / FRAME_MS)
    pre_roll_frames = int(PRE_ROLL_MS / FRAME_MS)
    min_frames = int(MIN_UTTERANCE_MS / FRAME_MS)
    max_frames = int(MAX_UTTERANCE_MS / FRAME_MS)

    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            # 你也可以改成 print(status) 觀察 underrun/overflow
            pass
        # indata shape (frames, channels)
        q.put(indata.copy())

    print(
        "=== VAD -> record -> Whisper API transcribe ===\n"
        f"SPEECHES_BASE_URL: {SPEECHES_BASE_URL}\n"
        f"MODEL: {TRANSCRIPTION_MODEL_ID}\n"
        f"Device: {DEVICE_INDEX if DEVICE_INDEX is not None else 'default'}\n"
        "Speak to trigger. Ctrl+C to stop.\n"
    )

    # pre-roll ring buffer (float32 frames)
    pre_roll: Deque[np.ndarray] = deque(maxlen=pre_roll_frames)

    recording = False
    voiced_frames: List[np.ndarray] = []
    silence_run = 0
    utter_frames = 0
    seg_count = 0

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=frame_samples,  # 讓 callback 每次剛好一個 frame
        dtype="float32",
        device=DEVICE_INDEX,
        callback=callback,
    ):
        while True:
            frame = q.get()  # shape (frame_samples, 1)
            pre_roll.append(frame)

            # VAD 判斷需要 PCM16 bytes 且 frame size 要符合 10/20/30ms
            pcm16 = pcm16_bytes_from_float32(frame)
            is_speech = vad.is_speech(pcm16, SAMPLE_RATE)

            if not recording:
                if is_speech:
                    recording = True
                    silence_run = 0
                    utter_frames = 0
                    voiced_frames = list(pre_roll)  # 先把 pre-roll 塞進去
                    print("🎤 ON")
                continue

            # recording == True
            voiced_frames.append(frame)
            utter_frames += 1

            if is_speech:
                silence_run = 0
            else:
                silence_run += 1

            # 句子結束條件：靜音一段時間 或 太長
            end_by_silence = silence_run >= silence_frames_end
            end_by_maxlen = utter_frames >= max_frames

            if end_by_silence or end_by_maxlen:
                recording = False
                print("🎤 OFF" + (" (maxlen)" if end_by_maxlen else ""))

                if utter_frames < min_frames:
                    # 太短，丟掉（通常是誤觸發）
                    print("  (skip: too short)")
                    voiced_frames.clear()
                    pre_roll.clear()
                    continue

                # 合併並存檔
                audio = np.concatenate(voiced_frames, axis=0)  # (N,1)
                ts = time.strftime("%Y%m%d_%H%M%S")
                seg_count += 1
                wav_path = os.path.join(OUTPUT_DIR, f"utt_{ts}_{seg_count:03d}.wav")
                save_wav(wav_path, audio, SAMPLE_RATE)
                print(f"  saved: {wav_path}")

                # 呼叫 Whisper API
                try:
                    resp = call_whisper_api(wav_path)
                    # 常見 key: text
                    # text = resp.get("text")
                    # if text is None:
                    #     # fallback: 印出 json
                    #     print("  transcription:", json.dumps(resp, ensure_ascii=False))
                    # else:
                    #     print("  transcription:", text)

                    text = resp.get("text")
                    if not text:
                        print("  transcription:", json.dumps(resp, ensure_ascii=False))
                    else:
                        print("  transcription:", text)

                        # ===== 呼叫 GPT =====
                        try:
                            reply = call_gpt_api(text)
                            print("🤖 GPT reply:", reply)
                        except requests.RequestException as e:
                            print("  GPT API error:", str(e))
                except requests.RequestException as e:
                    print("  API error:", str(e))

                # 重置
                voiced_frames.clear()
                pre_roll.clear()


if __name__ == "__main__":
    main()
