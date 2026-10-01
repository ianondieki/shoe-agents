"""The synthetic caller for the live call tests: Windows' own offline voice (free, no credits), played
into VB-Audio Virtual Cable (https://vb-audio.com/Cable/), a free virtual audio device - so a whole
call runs over real audio devices with nobody at the keyboard and nothing on the speakers."""
import os
import subprocess
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
CALLER_DIR = os.path.join(HERE, "_tmp", "caller")

LINES = {
    "hello": "Hi, this is Jane Doe.",
    "price": "How much is the cushioned trail runner?",
    "offer": "That's a bit steep. I'll give you eighty dollars.",
    "deal": "Okay, deal. I'll take it at that price.",
    "yes": "Yes please, go ahead and place the order.",
    "hold": "Actually, hold on a second.",
    "bye": "No, that's all. Thanks, bye!",
}

_SAPI = r'''param([string]$OutDir)
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice("Microsoft Zira Desktop")
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
foreach ($l in Get-Content -Path (Join-Path $OutDir "lines.txt")) {
  $name, $text = $l -split "\|", 2
  $s.SetOutputToWaveFile((Join-Path $OutDir "$name.wav"), $fmt)
  $s.Speak($text)
}
$s.SetOutputToNull()
'''


def caller_voice() -> dict[str, bytes]:
    """The caller's lines as 16 kHz mono PCM, spoken by Windows' offline voice. Made once, then reused."""
    os.makedirs(CALLER_DIR, exist_ok=True)
    missing = [k for k in LINES if not os.path.exists(os.path.join(CALLER_DIR, f"{k}.wav"))]
    if missing:
        with open(os.path.join(CALLER_DIR, "lines.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(f"{k}|{LINES[k]}" for k in missing))
        script = os.path.join(CALLER_DIR, "make.ps1")
        with open(script, "w", encoding="utf-8") as f:
            f.write(_SAPI)
        subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script,
                        "-OutDir", CALLER_DIR], check=True, capture_output=True)
    out = {}
    for k in LINES:
        with wave.open(os.path.join(CALLER_DIR, f"{k}.wav")) as w:
            out[k] = w.readframes(w.getnframes())
    return out


def cable() -> tuple[int, int]:
    """(speak_into, listen_on): the MME ends of the virtual cable - or stop, saying how to get one."""
    import sounddevice as sd

    def find(name, kind):
        for i, d in enumerate(sd.query_devices()):
            if (name.lower() in d["name"].lower() and d[f"max_{kind}_channels"] > 0
                    and sd.query_hostapis(d["hostapi"])["name"] == "MME"):
                return i
        raise SystemExit(f"No '{name}' device: install VB-Audio Virtual Cable (https://vb-audio.com/Cable/) "
                         "to run the live call tests.")

    return find("CABLE Input", "output"), find("CABLE Output", "input")
