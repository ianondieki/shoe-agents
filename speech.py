"""The speech half of a phone call, for the terminal: hearing the caller, and speaking as Mo.

Three pieces, each measured on this machine before it was written:
  Endpointer  - finds where one utterance starts and ends in a stream of microphone audio (webrtcvad).
  Transcriber - turns one utterance into text (Groq Whisper, ~0.5s warm).
  Voice       - turns Mo's text into audio in his ElevenLabs voice (~0.4s to first byte warm).

The brain - prices, floors, consent, orders - is not here. call.py drives voice_server's own
endpoint in-process, so a terminal call behaves exactly like a phone call through ElevenLabs.
"""
import asyncio
import hashlib
import io
import json
import os
import re
import wave
from collections import deque
from pathlib import Path

try:
    # Antivirus HTTPS scanning re-signs TLS on this machine; verify against the Windows store.
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import httpx
import numpy as np
import webrtcvad

MIC_RATE = 16000                 # what webrtcvad and Whisper both want
FRAME_MS = 30                    # webrtcvad accepts 10, 20 or 30 ms frames
FRAME_SAMPLES = MIC_RATE * FRAME_MS // 1000
VOICE_RATE = 24000               # the best raw-PCM rate the free ElevenLabs plan allows

GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
ELEVEN_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream"
MO_VOICE = os.getenv("MO_VOICE_ID", "iP95p4xoKVk53GoZ742B")     # "Chris - Charming, Down-to-Earth"
TTS_MODEL = os.getenv("TTS_MODEL", "eleven_flash_v2")          # English agents: flash v2
CACHE_DIR = Path(os.getenv("TTS_CACHE_DIR", Path(__file__).parent / ".cache" / "tts"))


# ---------- hearing: where does an utterance start and end? ----------
MIN_RMS = 60                     # about -55 dBFS: quieter than this is never a caller


def rms(frame: bytes) -> float:
    a = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(a * a))) if len(a) else 0.0


class Endpointer:
    """Feed 30 ms frames of 16 kHz mono int16; get back whole utterances.

    A frame is speech only if webrtcvad says so AND it is clearly louder than the room. webrtcvad
    alone called 66 of 66 frames of fan-like noise speech (measured), so an utterance would never
    end; Whisper then turns noise into confident text. The room's level is learned as the call goes
    - the quietest frames of the last three seconds - so a fan switching on is absorbed, not heard.

    Speech starts when most of the last few frames are speech (a click or a cough is not a caller)
    and ends when the last ~0.7 s hold almost none: a finished sentence, not a pause for breath.
    The 300 ms before the start are kept, so the first syllable is never clipped; only 300 ms of the
    closing silence are sent on (Whisper invents words in long silences).
    """

    def __init__(self, aggressiveness=2, start_frames=5, start_voiced=3, end_silence_ms=720,
                 end_tolerance=1, preroll_ms=300, keep_tail_ms=300, min_speech_ms=210,
                 max_utterance_s=25, above_floor=2.0):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.start_window = deque(maxlen=start_frames)
        self.start_voiced = start_voiced
        self.end_window = deque(maxlen=end_silence_ms // FRAME_MS)
        self.end_tolerance = end_tolerance
        self.preroll = deque(maxlen=preroll_ms // FRAME_MS)
        self.keep_tail = keep_tail_ms // FRAME_MS
        self.min_voiced = min_speech_ms // FRAME_MS
        self.max_frames = max_utterance_s * 1000 // FRAME_MS
        self.above_floor = above_floor
        self.levels: deque[float] = deque(maxlen=100)     # ~3 s of loudness: the room's floor
        self.reset()

    def reset(self):
        """Forget any half-heard utterance. The room's floor is kept: it is still the same room."""
        self.start_window.clear()
        self.end_window.clear()
        self.preroll.clear()
        self.frames: list[bytes] = []
        self.in_speech = False
        self.voiced = 0

    def prime(self, frames):
        """Learn the room from audio recorded while nobody was talking."""
        for f in frames:
            self.levels.append(rms(f))

    @property
    def floor(self) -> float:
        if len(self.levels) < 10:
            return 0.0
        return sorted(self.levels)[len(self.levels) // 10]

    def is_speech(self, frame: bytes) -> bool:
        loud = rms(frame)
        threshold = max(MIN_RMS, self.floor * self.above_floor)
        self.levels.append(loud)
        return loud > threshold and self.vad.is_speech(frame, MIC_RATE)

    def feed(self, frame: bytes) -> bytes | None:
        """One frame in. Returns the utterance's PCM when one has just ended, else None."""
        voiced = self.is_speech(frame)
        if not self.in_speech:
            self.preroll.append(frame)
            self.start_window.append(voiced)
            if sum(self.start_window) >= self.start_voiced:
                self.in_speech = True
                self.frames = list(self.preroll)
                self.voiced = sum(self.start_window)
                self.end_window.clear()
            return None

        self.frames.append(frame)
        self.voiced += voiced
        self.end_window.append(voiced)
        finished = (len(self.end_window) == self.end_window.maxlen
                    and sum(self.end_window) <= self.end_tolerance)
        if finished or len(self.frames) >= self.max_frames:
            drop = max(0, len(self.end_window) - self.keep_tail) if finished else 0
            pcm = b"".join(self.frames[:len(self.frames) - drop])
            enough = self.voiced >= self.min_voiced
            self.reset()
            # Too little real speech is noise that fooled the start detector: never transcribe it.
            return pcm if enough else None
        return None

    @property
    def speaking(self) -> bool:
        return self.in_speech


class EchoGuard:
    """Tells the caller's voice from Mo's own, coming back into the microphone.

    The microphone stays open while Mo talks, so the caller can cut in by simply talking - but on a
    laptop every word Mo says goes straight back into it. This keeps the loudness of what the speaker
    is playing, moment by moment, and counts a frame as the caller only when it is louder than that
    echo can explain. How much of Mo really comes back is measured as the call goes (headphones: next
    to nothing; a laptop's speakers: a good share of him), starting from the worst case. In the gaps
    between his words nothing is expected back at all, which is where a caller who starts talking is
    heard first.
    """

    def __init__(self, margin=1.8, sustain_ms=150, delay_ms=60, coupling=1.0, keep_ms=600):
        self.margin, self.coupling = margin, coupling
        self.sustain = max(1, sustain_ms // FRAME_MS)
        self.delay = delay_ms / 1000
        self.playing: deque[tuple[float, float]] = deque(maxlen=600)   # (heard at, loudness)
        self.ratios: deque[float] = deque(maxlen=400)
        self.recent: deque[bytes] = deque(maxlen=max(1, keep_ms // FRAME_MS))
        self.run = 0

    def plays(self, audio: np.ndarray, rate: int, at: float):
        """One of Mo's clips, and when its first sample reaches the speaker."""
        step = max(1, rate * FRAME_MS // 1000)
        a = np.asarray(audio, dtype=np.float32)
        for i in range(0, len(a) - step + 1, step):
            chunk = a[i:i + step]
            self.playing.append((at + i / rate, float(np.sqrt(np.mean(chunk * chunk)))))

    def silence(self, after: float):
        """Mo was cut off: whatever was still queued never played, so none of it comes back."""
        while self.playing and self.playing[-1][0] > after:
            self.playing.pop()

    def echo(self, when: float) -> float:
        """The loudest thing playing around the moment this frame was captured - a few milliseconds
        of misalignment must not look like a caller."""
        wanted, near = when - self.delay, 2 * FRAME_MS / 1000
        return max((loud for at, loud in self.playing if abs(at - wanted) <= near), default=0.0)

    def hears_caller(self, frame: bytes, when: float, ear: "Endpointer") -> bool:
        """One frame from the open microphone while Mo talks. True once the caller has cut in."""
        self.recent.append(frame)
        loud, playing = rms(frame), self.echo(when)
        if playing > 40 and loud > 0:
            self.ratios.append(loud / playing)      # how much of Mo comes back, while he is loud
        floor = max(MIN_RMS, ear.floor * ear.above_floor)
        over = loud > max(floor, playing * self.coupling * self.margin)
        self.run = self.run + 1 if over and ear.vad.is_speech(frame, MIC_RATE) else 0
        return self.run >= self.sustain

    def learn(self, clean: bool = True):
        """After one of Mo's turns: how much of him comes back at its WORST - the typical case is no
        use, because the loudest moments are what would stop him for nothing. It starts at the worst
        case there is (all of him) and settles down only over turns nobody talked over: a turn that
        was cut into teaches nothing, since the caller's voice is in the numbers."""
        if clean and len(self.ratios) >= 20:
            worst = float(np.quantile(np.array(self.ratios), 0.9))
            self.coupling = float(min(1.5, max(0.02, max(0.35 * self.coupling, worst))))
        self.ratios.clear()
        self.run = 0

    def false_cut(self):
        """Mo was stopped and there were no words in it: that was his own voice. Expect more back."""
        self.coupling = float(min(1.5, self.coupling * 1.6))
        self.ratios.clear()

    def first_words(self) -> list[bytes]:
        """The frames going by as the caller cut in: their first words are in here, over Mo's voice.
        Worth keeping more than the moment of the cut - where Mo's own voice is loud (a laptop on a
        hard desk), it can take him most of a second to be sure it is the caller."""
        words = list(self.recent)
        self.recent.clear()
        self.run = 0
        return words


def frames_of(pcm: bytes):
    """Split PCM into whole 30 ms frames (the trailing partial frame is dropped)."""
    step = FRAME_SAMPLES * 2
    for i in range(0, len(pcm) - step + 1, step):
        yield pcm[i:i + step]


def to_wav(pcm: bytes, rate: int = MIC_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


# ---------- hearing: what did they say? ----------
def louder(pcm: bytes) -> bytes:
    """Lift a faint recording so its peak sits at -6 dBFS (at most x20): Whisper hears quiet
    microphones measurably worse, and a laptop mic at arm's length is quiet."""
    a = np.frombuffer(pcm, dtype=np.int16)
    peak = int(np.abs(a.astype(np.int32)).max()) if len(a) else 0
    if not peak or peak >= 3277:                       # already louder than -20 dBFS
        return pcm
    gain = min(16422 / peak, 20.0)
    return np.clip(a.astype(np.float32) * gain, -32768, 32767).astype(np.int16).tobytes()


# Whisper's famous phantom lines, learned from subtitle data. Real callers do say "thank you" and
# "bye", so those are NOT here - the speech detector keeps silence away from Whisper instead.
_PHANTOMS = re.compile(r"thanks? for watching|subscribe|subtitles?|amara\.org|www\.|\[music\]|\(music\)",
                       re.I)


class Transcriber:
    home = "https://api.groq.com"

    def __init__(self, api_key: str | None = None, vocabulary: str = ""):
        self.key = api_key or os.environ.get("GROQ_API_KEY", "")
        # Whisper takes a prompt that biases what it expects to hear: names and prices from the
        # shop itself turn a bare "Eighty" (heard as "Eli" on a real call) back into a number.
        self.vocabulary = vocabulary
        self.client = httpx.AsyncClient(timeout=20)

    async def text(self, pcm: bytes) -> str:
        """Microphone audio from a phone call: 16 kHz mono PCM, lifted if it came in faint."""
        return await self.spoken(to_wav(louder(pcm)), "utterance.wav", "audio/wav")

    async def spoken(self, data: bytes, filename: str = "clip.webm", mime: str = "audio/webm") -> str:
        """An audio file as a browser or WhatsApp hands it over: webm, ogg, mp4, m4a or wav.

        Whisper takes the container as it comes, so nothing has to be transcoded on this machine.
        """
        if not self.key:
            raise SpeechUnavailable("No GROQ_API_KEY in .env, so Mo can't hear you. "
                                    "Get a free key at console.groq.com/keys.")
        # "json", not "verbose_json": Groq's no_speech_prob came back 0 even for the "Thank you." Whisper
        # invents on pure silence, and that phantom scored a better avg_logprob than a real "Yes." - so
        # noise is kept away from Whisper by the Endpointer, not filtered by a confidence score after.
        form = {"model": STT_MODEL, "language": "en", "temperature": "0", "response_format": "json"}
        if self.vocabulary:
            form["prompt"] = self.vocabulary
        for attempt in (1, 2):
            r = await self.client.post(GROQ_STT_URL, headers={"Authorization": f"Bearer {self.key}"},
                                       files={"file": (filename, data, mime)}, data=form)
            if r.status_code != 429 or attempt == 2:
                break
            # The free tier allows 20 utterances a minute; wait as long as Groq asks, within reason.
            await asyncio.sleep(min(float(r.headers.get("retry-after") or 2), 5))
        if r.status_code == 429:
            raise SpeechUnavailable("Speech recognition is busy (Groq's free limit) - say that again in a moment.",
                                    lasting=False)
        if r.status_code == 401:
            raise SpeechUnavailable("Groq refused the GROQ_API_KEY in .env - check it at console.groq.com/keys.")
        r.raise_for_status()
        said = (r.json().get("text") or "").strip()
        if not re.search(r"[A-Za-z0-9]", said) or _PHANTOMS.search(said):
            return ""
        # The vocabulary prompt can leak back verbatim on unclear audio: a long run of it is not speech.
        # A short one is - "Jane Doe." and "Eighty." are the very answers the prompt is there to catch.
        echo = said.lower().strip(" .")
        if self.vocabulary and len(echo.split()) >= 5 and echo in self.vocabulary.lower():
            return ""
        return said

    async def aclose(self):
        await self.client.aclose()


# ---------- speaking: Mo's voice ----------
class SpeechUnavailable(Exception):
    """A speech service is out of reach. The call goes on either way.

    lasting=True: it will not come back this call (no key, no credits, no permission) - stop trying.
    lasting=False: a hiccup (a timeout, a busy server) - skip this sentence and try the next one.
    """

    def __init__(self, message: str, lasting: bool = True):
        super().__init__(message)
        self.lasting = lasting


class Voice:
    """Mo's words to 24 kHz mono int16 audio. Fixed lines are cached on disk and cost nothing again."""
    home = "https://api.elevenlabs.io"

    def __init__(self, api_key: str | None = None, voice_id: str = MO_VOICE, model: str = TTS_MODEL):
        self.key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        self.voice_id, self.model = voice_id, model
        self.client = httpx.AsyncClient(timeout=20)
        self.chars_used = 0

    def _cache_path(self, text: str) -> Path:
        h = hashlib.sha1(f"{self.voice_id}|{self.model}|{VOICE_RATE}|{text}".encode()).hexdigest()[:20]
        return CACHE_DIR / f"{h}.pcm"

    async def speak(self, text: str, cache: bool = False, fmt: str = ""):
        """Mo saying one line. Raw 16-bit PCM by default, for the speakers; `fmt` asks ElevenLabs for
        a container instead ("mp3_44100_64" for a web page or a WhatsApp voice note), which comes
        back as bytes rather than samples."""
        path = self._cache_path(text + fmt)
        if cache and path.exists():
            kept = path.read_bytes()
            return kept if fmt else np.frombuffer(kept, dtype=np.int16)
        if not self.key:
            raise SpeechUnavailable("No ELEVENLABS_API_KEY in .env - Mo will reply in text.")
        for attempt in range(3):
            pcm = bytearray()
            try:
                async with self.client.stream(
                    "POST", ELEVEN_TTS_URL.format(voice=self.voice_id),
                    params={"output_format": fmt or f"pcm_{VOICE_RATE}",
                            "optimize_streaming_latency": 3},
                    headers={"xi-api-key": self.key}, json={"text": text, "model_id": self.model},
                ) as r:
                    if r.status_code == 429 and attempt < 2:
                        # The free plan makes only a few voices at once; this is a queue, not a failure.
                        await asyncio.sleep(0.6 * (attempt + 1))
                        continue
                    if r.status_code >= 400:
                        raise SpeechUnavailable(*_voice_problem(r.status_code, await r.aread()))
                    async for chunk in r.aiter_bytes():
                        pcm.extend(chunk)
                break
            except httpx.HTTPError as e:
                raise SpeechUnavailable(f"Mo's voice dropped out for a moment ({type(e).__name__}).",
                                        lasting=False) from e
        self.chars_used += len(text)
        if len(pcm) % 2:
            pcm = pcm[:-1]
        if cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes(pcm))
        if fmt:
            return bytes(pcm)                 # an mp3 (or whatever was asked for), not samples
        return np.frombuffer(bytes(pcm), dtype=np.int16)

    async def credits_left(self) -> tuple[int, int] | None:
        """(used, limit) of this month's ElevenLabs characters, or None if it cannot be read."""
        try:
            r = await self.client.get("https://api.elevenlabs.io/v1/user/subscription",
                                      headers={"xi-api-key": self.key})
            j = r.json()
            return j["character_count"], j["character_limit"]
        except Exception:
            return None

    async def aclose(self):
        await self.client.aclose()


def _voice_problem(status: int, body: bytes) -> tuple[str, bool]:
    """ElevenLabs' error, in words a caller can act on - and whether it will last the call."""
    try:
        detail = json.loads(body).get("detail")
        code = (detail.get("status") if isinstance(detail, dict) else "") or ""
    except Exception:
        code = ""
    text = body.decode(errors="replace").lower()
    if code == "quota_exceeded" or "quota" in text:
        return "ElevenLabs voice credits are used up for this month - Mo will reply in text.", True
    if code == "missing_permissions":
        return ("Your ElevenLabs key isn't allowed to make speech: enable 'Text to Speech' for the key "
                "(elevenlabs.io > Developers > API keys). Mo will reply in text."), True
    if code == "detected_unusual_activity" or "unusual activity" in text:
        return "ElevenLabs has paused free use on this account ('unusual activity'). Mo will reply in text.", True
    if status == 401:
        return "ElevenLabs refused the ELEVENLABS_API_KEY in .env - Mo will reply in text.", True
    if status in (403, 404):
        return f"Mo's voice is not available to this key (ElevenLabs HTTP {status}) - he will reply in text.", True
    return f"Mo's voice dropped out for a moment (ElevenLabs HTTP {status}).", False


def shop_vocabulary() -> str:
    """Names, shoes and money words from the shop's own database, as a Whisper prompt."""
    from db import db  # imported late: DB_PATH must already point at the right database

    with db() as conn:
        names = [r[0] for r in conn.execute("SELECT CustomerName FROM CustomerInfo")]
        shoes = [r[0] for r in conn.execute("SELECT StyleDesc FROM ShoeInventory")]
    return ("A phone call to Mo's shoe shop. " + ", ".join(names) + ". " + ", ".join(shoes)
            + ". Eighty dollars, ninety dollars, a hundred and eight dollars, one twenty.")


async def warm(*parts):
    """Open the TLS connections before the caller speaks: the first request on a cold connection
    measured 1.6-1.9 s, a warm one 0.4-0.6 s. Takes Transcribers and Voices (anything with .home)."""
    await asyncio.gather(*(p.client.head(p.home) for p in parts if hasattr(p, "home")),
                         return_exceptions=True)
