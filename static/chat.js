/* Talking to Mo in a browser.
   Text goes up and his sentences come back one at a time, so the first lands while he is still
   working on the rest. On the call page the same turn starts as a recording instead, and each
   sentence is spoken as it arrives - ElevenLabs while the shop has credits, the browser's own voice
   when it does not, so a call never ends in silence. */
(function () {
  const thread = document.getElementById("thread");
  const form = document.getElementById("ask");
  const box = document.getElementById("text");
  const mic = document.getElementById("mic");
  const speaks = window.MO && window.MO.voice;
  let busy = false;

  function add(who, text, extra) {
    const p = document.createElement("p");
    p.className = "said " + who + (extra ? " " + extra : "");
    p.textContent = text;
    thread.appendChild(p);
    window.scrollTo(0, document.body.scrollHeight);
    return p;
  }

  function orderLink(id) {
    const p = add("mo", "");
    const a = document.createElement("a");
    a.href = "/orders/" + id;
    a.className = "button pay stamp";
    a.textContent = "Pay order " + id + " with M-Pesa";
    p.appendChild(a);
  }

  /* One sentence in Mo's voice. The page waits for it to finish, so he is never two sentences ahead
     of himself, and falls back to the browser's voice when the shop is out of credits. */
  async function say(line) {
    if (!speaks) return;
    try {
      const r = await fetch("/api/tts?text=" + encodeURIComponent(line));
      if (r.ok && r.status !== 204) {
        const sound = new Audio(URL.createObjectURL(await r.blob()));
        await new Promise((done) => { sound.onended = sound.onerror = done; sound.play().catch(done); });
        return;
      }
    } catch (e) { /* fall through to the browser's own voice */ }
    if (window.speechSynthesis) {
      await new Promise((done) => {
        const said = new SpeechSynthesisUtterance(line);
        said.onend = said.onerror = done;
        speechSynthesis.speak(said);
      });
    }
  }

  async function send(text) {
    if (!text || busy) return;
    busy = true;
    const chips = document.getElementById("openers");
    if (chips) chips.remove();        // said once; the conversation takes over from here
    add("you", text);
    box.value = "";
    const waiting = add("mo", "...", "note");
    try {
      const r = await fetch("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
      });
      if (r.status === 429) { waiting.textContent = "Too many messages at once. Give it a moment."; return; }
      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "", first = true;
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop();
        for (const part of parts) {
          const line = part.replace(/^data: /, "").trim();
          if (!line || line === "[DONE]") continue;
          const event = JSON.parse(line);
          if (event.say) {
            if (first) { waiting.remove(); first = false; }
            add("mo", event.say);
            await say(event.say);
          } else if (event.done) {
            if (first) { waiting.remove(); first = false; }
            if (event.done.ordered && event.done.order_id) orderLink(event.done.order_id);
            if (event.done.error) add("mo", "Mo's line dropped for a second. Say that again?", "note");
          }
        }
      }
    } catch (e) {
      waiting.textContent = "That did not reach the shop. Check your connection and try again.";
    } finally {
      busy = false;
    }
  }

  form.addEventListener("submit", function (e) { e.preventDefault(); send(box.value.trim()); });

  /* Three ways in, for anyone looking at an empty thread wondering what to say. They go once: after
     the first message the conversation itself is the invitation. */
  const openers = document.getElementById("openers");
  if (openers) {
    openers.addEventListener("click", function (e) {
      const chip = e.target.closest("button");
      if (!chip || busy) return;
      send(chip.textContent.trim());
    });
  }

  /* Hold to talk. Whatever the phone records natively - webm on Android, mp4 on iOS - is what
     Whisper is handed; nothing is converted on the way. */
  if (mic) {
    let recorder = null, chunks = [], kind = "audio/webm";
    const start = async function (e) {
      e.preventDefault();
      if (busy || recorder) return;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        recorder = new MediaRecorder(stream);
        kind = recorder.mimeType || "audio/webm";
        chunks = [];
        recorder.ondataavailable = (ev) => chunks.push(ev.data);
        recorder.onstop = async function () {
          stream.getTracks().forEach((t) => t.stop());
          const clip = new Blob(chunks, { type: kind });
          recorder = null;
          if (clip.size < 1200) return;                       // a tap, not a sentence
          busy = true;
          const waiting = add("mo", "...", "note");
          const body = new FormData();
          body.append("audio", clip, "clip." + (kind.indexOf("mp4") >= 0 ? "mp4" : "webm"));
          try {
            const r = await fetch("/api/voice", { method: "POST", body: body });
            const out = await r.json();
            waiting.remove();
            if (out.heard) add("you", out.heard);
            if (out.error) add("mo", out.error, "note");
            for (const line of out.say || []) { add("mo", line); await say(line); }
            if (out.done && out.done.ordered && out.done.order_id) orderLink(out.done.order_id);
          } catch (err) {
            waiting.textContent = "That recording did not get through. Try again.";
          } finally {
            busy = false;
          }
        };
        recorder.start();
        mic.dataset.on = "1";
      } catch (err) {
        add("mo", "This browser will not give the page a microphone. Type instead.", "note");
      }
    };
    const stop = function (e) {
      e.preventDefault();
      mic.dataset.on = "";
      if (recorder && recorder.state === "recording") recorder.stop();
    };
    mic.addEventListener("pointerdown", start);
    mic.addEventListener("pointerup", stop);
    mic.addEventListener("pointercancel", stop);
    mic.addEventListener("pointerleave", stop);
  }
})();
