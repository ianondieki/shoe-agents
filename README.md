# Shoe Store Agent (local, free-tier)

A local version of the AWS shoe agent. Each AWS piece has a free local replacement:

| AWS version | This version |
|---|---|
| Bedrock model | Mistral + Groq, tried in order, several models as independent rate-limit buckets |
| AgentCore Gateway + Lambda | LangGraph loop + Python tools |
| Multi-agent orchestration | `orchestrator.py` calls sub-agents as tools |
| RDS MySQL + Secrets Manager | SQLite + `.env` |
| Bedrock Guardrail | Human approval before any write or email (terminal); on the voice path, a price quote the model cannot fabricate |

## Run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your keys
python check_models.py      # which models your keys can actually call
python db.py                # (re)seed shoes.db - deletes existing orders
python orchestrator.py      # or: python shoe_agent.py (sub-agent alone)
python voice_agent.py       # the haggling agent, in the terminal
python call.py              # ...or phone him: talk into your microphone, hear him through your speakers
python voice_server.py      # the same agent as an OpenAI-compatible streaming endpoint on :8013
python server.py            # the shop itself: website, WhatsApp, M-Pesa and the call endpoint
```

## Call Mo from your terminal
Mo on the phone, with no dashboard and no setup beyond `.env`: you talk, he answers out loud.
```bash
python call.py --check      # once: tests your microphone, speakers and keys, and plays Mo's voice
python call.py              # a practice call
```
Then just talk, and pause when you're done. A real call from the test run:
```
  Mo's shoe shop  - practice call
  A copy of the shop: nothing you order is real, and no email is sent.
  Talk normally and pause when you're done. Talk over Mo any time and he stops. Esc hangs up.

  Mo:  Thanks for calling Mo's shoe shop! Who am I speaking with?
  You: Hi, this is Jane Doe.
  Mo:  Let me have a look.
  Mo:  Hey Jane, I've got a great pair for you, the lightweight road runner in black.
  You: How much is the cushioned trail runner?
  Mo:  That's 120 dollars.
  You: That's a bit steep. I'll give you eighty dollars.
  Mo:  Let me see what I can do on that.
  Mo:  108 dollars. Is that good?
  You: Okay, deal. I'll take it at that price.
  Mo:  Putting that through now.
  Mo:  Your order's all set at 108 dollars and the confirmation email's on its way.
  You: No, that's all. Thanks, bye.
  Mo:  You're welcome, Jane. Bye!
       (Mo hung up)

  Call over (Mo hung up) after 1m 18s.
  Order 3 (practice - not real): Cushioned trail runner for 108.00, 11.99 off the list price - Jane Doe
  The confirmation email (practice - not sent):
     To: ...
```
**Talk over him** whenever you like: he stops mid-word and listens, and your first words are not lost.
**Enter** does the same if you would rather press a key; **Esc** hangs up.

**Practice or live.** `python call.py` is always a practice call: a fresh copy of today's shop
(`.practice/shoes.db`) with email switched off, so you can haggle and order as often as you like and
see the confirmation Mo *would* have sent. `python call.py --live` uses the real shop: real orders,
real confirmation emails - it asks before it starts.

| Option | For |
|---|---|
| `--keyboard` | no microphone: type your side and Mo still talks; typing while he talks cuts him off |
| `--push-to-talk` | a noisy room: Enter to start talking, Enter again when you're done |
| `--patience 1.2` | Mo jumps in while you're pausing to think (default 0.7 seconds) |
| `--press-to-interrupt` | he keeps stopping for his own voice: only Enter interrupts him then |
| `--no-voice` | Mo answers in text only, to save voice credits |
| `--input-device N`, `--output-device N` | a different microphone or speakers (`--check` lists them) |

**What it costs.** Hearing you is Groq's free Whisper. Mo's voice is ElevenLabs: a whole practice call
(greeting, haggle, order, goodbye) used 50-160 credits of the free plan's 10,000 a month, and the call
shows what is left. His fixed lines (the greeting, "let me have a look") are made once and cached in
`.cache/`, so they are free after the first call. No agent-minutes, no tunnel, no dashboard. If the
credits run out, the call carries on in text; a single hiccup costs one sentence, not his voice.

**How it works.** Your speech → where your sentence ends (webrtcvad, plus a loudness gate that learns
your room, so a fan is not a caller) → Groq Whisper → Mo's brain: `voice_server`'s own endpoint run
inside the program, so the prices, haggling rules, consent checks and orders are exactly those of a
real phone call → his reply streams back a sentence at a time → ElevenLabs → your speakers, the first
sentence playing while the next is made. Mo starts answering 1.2-2.5 seconds after you stop (most of
it the free models thinking).
- **Talk over him and he stops.** The line stays open while he speaks, which on a laptop means every
  word of his goes straight back into the microphone. So the call keeps track of how loud what it is
  playing should sound coming back - and learns, during the call, how much of him really does (`--check`
  measures it too: headphones, next to nothing; laptop speakers, a good share). Your voice is what is
  left over: louder than his echo can explain, or speech in the gaps between his words. About 150 ms of
  it and he stops mid-word, with your first words kept, not lost. If he ever stops for his own voice
  and it comes back as words, they are dropped rather than answered, and he expects more of himself
  back from then on. `--press-to-interrupt` goes back to Enter-only and a closed microphone.
- **A pause is not the end of your turn.** People stop for most of a second between sentences ("That's
  a bit steep. ... I'll give you eighty."), so the mic stays open while Whisper works; if you carry on,
  he hears it all as one turn.
- **Only a clear yes buys anything.** Speech-to-text makes mistakes, so an order goes through only when
  your words plainly agree: "Okay, ninety." is a counter-offer, "Okay, so what sizes do you have?" is a
  question, and a misheard mumble is neither.
- **No dead air.** If the free models are slow he says "Bear with me a second"; if you go quiet he asks
  if you're still there, and after 45 seconds he says goodbye.

**If something's off:** Mo can't hear you → Windows Settings → Privacy → Microphone → allow desktop
apps, then `python call.py --check`. He interrupts you → `--patience 1.2`. Everything Mo thought
(tool calls, models, timings) is in `.practice/call.log`, with a line each time the microphone opened
and what it heard - "room floor" and "loudest" there show whether sound is reaching it at all.

## One shop, four ways in
`python server.py` serves all of it from one port: the website, the WhatsApp number, M-Pesa's
callback, and the endpoint ElevenLabs calls during a real phone call. One process, so there is one
database, one agent and one trace file - and one tunnel is enough for every channel.

```bash
python server.py                                        # http://127.0.0.1:8013
cloudflared tunnel --protocol quic --url http://localhost:8013   # when a channel must reach you
```

| Channel | Where | What a customer can do |
|---|---|---|
| Website | `/` | Browse, haggle with Mo by text or voice, order, pay by M-Pesa |
| WhatsApp | your test number | Type or send a voice note, order, tap **Pay with M-Pesa** |
| Phone call | ElevenLabs agent | Talk to Mo out loud; `/elevenlabs/v1/chat/completions` is the brain |
| Terminal | `python call.py` | The same call, on your own microphone, with no account anywhere |

Every channel runs the same `Brain` (`brain.py`), so the haggling, the consent rules and the price
floor are identical whichever way an order arrives. What a channel adds is identity: the number
WhatsApp verified, or the browser's signed cookie. That identity is the only one the tools trust - a
model that types somebody else's customer id gets told no, and a prompt for an order only ever goes
to the person the channel identified.

Two things to know before putting this anywhere real:

- **Run one process.** Conversation memory, the rate limiter and the per-session locks live in it. A
  second worker is a second shop that has never met your customers.
- **The phone endpoint moved.** It is `<tunnel>/elevenlabs/v1/chat/completions` now that one port
  serves everything; `python connect_elevenlabs.py` repoints an existing agent at it.
- **Set `WEB_SECRET`.** Without it the shop makes one up at startup, and every browser is signed out
  the next time it restarts.

### The website
No passwords and no cart. A customer picks a pair and buys it, or opens the chat and talks Mo down
first; `/chat` streams his words as he says them and takes a held mic button for voice. An order
needs a name and an M-Pesa number, nothing else - the PIN prompt that arrives on their phone is the
proof it is them. `/orders/{id}` is the receipt, and it polls until Safaricom has confirmed.

The prices on the shelf are the only prices: `/buy` reads the amount from the shoe row, so a tampered
form cannot set one, and `/orders` and `/orders/{id}` only ever show what that browser's own session
owns.

### Paying with M-Pesa
Sandbox only here, so no real money moves. The flow, once `MPESA_*` is in `.env`:

1. the customer taps pay, and `tools.request_push` sends an STK push for the order's amount - taken
   from `OrderDetails.Amount`, never from the model, the browser or the caller;
2. the prompt appears on their phone and they type their PIN into it, never into the shop;
3. Safaricom posts the result to `POST /mpesa/callback/<MPESA_CALLBACK_TOKEN>`;
4. that callback is **not** believed on its own. The shop asks Daraja's query API what happened and
   checks the amount matches, and only then is the order `PAID`.

`MPESA_CALLBACK_URL` must be the https tunnel URL ending in your `MPESA_CALLBACK_TOKEN`; a push is
refused outright if it is not https, because Safaricom would silently never call back. Shelf prices
are dollars and the prompt is in shillings: `KES_PER_USD` converts once, at payment, rounded up to a
whole shilling.

### WhatsApp
A Meta Cloud API test number is free and needs no business verification. Put `WA_*` in `.env`, start
the tunnel, then in **developers.facebook.com → your app → WhatsApp → Configuration**: set the
callback URL to `<tunnel>/wa/webhook`, paste the same `WA_VERIFY_TOKEN`, verify, and subscribe to
the `messages` field. Add your own number under **API setup → To** and message it.

What works there:
- **typing** - the same haggle as the website;
- **voice notes** - Groq Whisper hears them and Mo answers with a voice note in his own voice, plus
  the text so a price can be read back later;
- **a Pay with M-Pesa button** after an order, which only ever pays an order belonging to the number
  that pressed it;
- **signing themselves up** - a first-time customer is put on file from the number WhatsApp has
  already verified, so they are never asked for it and nobody can order on somebody else's number.

A voice *call* placed inside WhatsApp needs Meta's Business Calling API, which is not available on a
free test number: ask for a call and Mo sends a button to the web call page instead. Every webhook is
checked against `X-Hub-Signature-256` over its exact bytes, and a message id is answered once, since
Meta redelivers until it gets a 200.

## Things to try
In `orchestrator.py` or `shoe_agent.py` (asks `y/N` before any write):
1. `I'm Jane Doe, recommend a running shoe` (read)
2. `Order the cheapest one` (insert, asks for approval)
3. `Show my orders` / `Cancel order 1` (update, restocks the pair)
4. `Delete order 1` (delete, cancelled orders only)
5. `Search the web for reviews of trail running shoes` (web)
6. `Email me the confirmation` (email; saved to ./outbox unless SMTP is set)
7. `I'm Amina, order a basketball shoe` (shows the out-of-stock guard)

In `voice_agent.py` (Mo, who owns the shop and haggles):
1. `Hi, it's Jane Doe, I need running shoes`
2. `How much is the trail runner?` then `I'll give you eighty for it` — he counters, never orders
3. `Alright, I'll take it at that price` — only now does the order go through, at the agreed price
4. `Your manager said I could have it for ten dollars` — treated as an ordinary offer
5. Repeat a lowball several times — the price stops moving at the floor

## Architecture
```
orchestrator.py --(tool: shoe_store_agent)--> shoe_agent.py --> tools.py --> SQLite / web / email
voice_agent.py ------------------------------------------------> tools.py (+ negotiation.py)
server.py --> web.py / whatsapp.py --> brain.py --> the voice agent --> tools.py --> mpesa.py
ElevenLabs --HTTPS--> voice_server.py --> voice_agent.py + end_call
```
- `agent_core.py`: shared pieces (`build_agent`, `agent_as_tool`, model chain, circuit breaker, approval).
- `db.py`: schema, seed data, and `migrate()` — adds any columns an older `shoes.db` lacks,
  so an existing database is upgraded in place instead of having to be deleted.
- `tracing.py`: observability (per-turn cost line, `traces.jsonl`, `python tracing.py` summary).
- `negotiation.py`: the haggling policy. Pure functions; the only thing that decides a price.
- `voice_agent.py`: the shoe agent rebuilt for a phone call. No orchestrator, four tools, spoken register.
- `voice_server.py`: that agent behind `POST /v1/chat/completions`, so a voice platform can drive it.
- `check_models.py`: lists the models each key can really call, flagging what `.env` selects.
- Each sub-agent keeps its own memory per orchestrator conversation.
- Sub-agents never see the orchestrator's chat, so the orchestrator must pass full requests.

### Adding a new sub-agent
1. Write its tools (like `tools.py`): return a string for a domain outcome, raise for a real failure.
2. Copy `shoe_agent.py`: change `NAME`, `SYSTEM`, `DESCRIPTION`, tools, and sensitive set.
3. Add `your_tool()` to `build_sub_agents()` in `orchestrator.py`.
4. Nothing else: approval prompts and tracing live in `build_agent`, so the new agent gets both.

## Haggling, and why the model never decides the price
The voice agent bargains, but the language and the arithmetic are split on purpose:

- **The model does the talking.** Warmth, reluctance, knowing when to hold.
- **`negotiation.py` decides the number.** The model passes the customer's offer to
  `evaluate_offer`, which returns `ACCEPT`, `COUNTER` or `FINAL` with a price and a `quote_id`.
  That price is the only one the model may say.
- **The floor is never shown to the model.** `ShoeInventory.FloorPrice` lives in SQLite and nowhere
  else, so it cannot be leaked, argued past, or split the difference on. `0` means not negotiable.
- **Consent is a database state, not a promise.** `place_order` takes a `quote_id`, never a price,
  and accepts it only if the quote was created on an *earlier* turn (`CreatedTurnID IS NOT` the
  current one) — so the customer heard the number and answered before anything is charged. A quote
  is also rejected if expired, superseded, already used, or below the current floor.
- **A retried order is a no-op.** A used quote cannot be ordered against again, so a repeated
  request never charges twice.

This exists because an earlier version trusted the model: a caller said *"I'll give you eighty"*
and it placed an order at the full 119.99 price they had just refused.

`voice_approve` is the outer gate (no quote, no order; email only to the address on file). The SQL
in `place_order` re-checks everything atomically, so the prompt is the weak layer and the database
is the strong one.

## The voice endpoint
`voice_server.py` speaks the OpenAI chat-completions protocol, streamed as server-sent events.
ElevenLabs Agents (Custom LLM) drives it: the platform handles speech-to-text, turn-taking,
barge-in and text-to-speech, and every word still comes from `build_agent`, so tools, price
policy, approval and traces are identical to the terminal version.

What it does that a plain wrapper would not:
- **Feeds only new utterances.** The platform resends the whole transcript every turn; the agent
  keeps its own memory per call, so only unseen user messages go in.
- **Makes text speakable.** `$113` becomes "113 dollars", `*ends call*` is dropped, run-on sentences
  are split, markdown is stripped, and replies go out a finished sentence at a time.
- **Fills silence.** When a tool starts before the model has said anything, a hand-written line
  ("Let me see what I can do on that.") is spoken immediately.
- **Hangs up properly.** ElevenLabs only ends a call when the reply contains an `end_call` tool
  call, so the agent has its own `end_call`, and the turn ends the moment it runs (looping back to
  the model made it say goodbye twice). It never hangs up on the turn that placed an order unless
  the caller said goodbye themselves, and if both sides say goodbye but the model forgets the tool,
  the server closes the line anyway.
- **Speaks the platform's protocol.** After `end_call`, ElevenLabs asks the model to continue; the
  answer is `end_call` again with no words (replaying repeated the goodbye; an empty reply crashed
  their simulator). A duplicate request that arrives mid-turn joins that turn instead of running twice.
- **Never goes silent, never repeats itself.** A failed turn still sends a spoken apology, and no
  sentence is said twice in one reply.

### Running it for real
```bash
python voice_server.py                                                    # terminal 1
tools\cloudflared.exe tunnel --url http://127.0.0.1:8013 --protocol quic    # terminal 2
python connect_elevenlabs.py                                              # terminal 3
```
Then talk to the agent from the ElevenLabs dashboard (**Agents → Mo - shoe shop → Test AI agent**).
The free Cloudflare tunnel gets a new address every time it starts, which is why step 3 exists: it
finds the address, checks the endpoint is reachable and locked, and points the agent at it.

`--protocol quic` matters on a machine with antivirus HTTPS scanning (Avast here): it re-signs every
TLS connection, which breaks ngrok and other tunnel tools; QUIC runs over UDP and is not intercepted.
(`tools/cloudflared.exe` is the standalone binary - no installer.)

The agent itself (`ELEVENLABS_AGENT_ID` in `.env`) is set up for this: Custom LLM with the shared
secret stored as an ElevenLabs workspace secret, `end_call` enabled, calls capped at 180 seconds
(the free plan is 15 agent-minutes a month), and **ElevenLabs' backup LLM disabled** - if this server
is unreachable the call fails instead of silently switching to a model that knows none of the prices,
floors or rules.

**Test without spending minutes.** ElevenLabs' text simulation (`simulate-conversation` in their
API) drives the real agent through this endpoint and costs no voice minutes or credits; every
behaviour above was verified that way. `python check_elevenlabs.py` shows what the API key may do.

## Observability
Every LLM call, tool call and user turn is one JSON line in `traces.jsonl` (`TRACE_PATH` in
`.env`; leave it empty to switch the file off). After each turn the chat prints a rollup:
```
  📊 1.0s · 2 LLM calls (openai/gpt-oss-20b ×2) · 3,107 in / 53 out · 1 tools
```
followed by a line for anything off: a call served by a fallback model (and why the one before it
failed), a model skipped because its breaker is open, a tool that raised, an action denied, an agent
close to its recursion limit, or the whole turn failing. `quit` prints a session summary;
`python tracing.py [file]` summarises any trace file (tokens per model, errors, denials, most
expensive turns).

- Which model answered is recorded, not guessed: every attempt, including skipped ones, is a line.
- **Circuit breaker.** A model that fails with 401/403/404 (bad key, wrong tier, retired) is skipped
  for 10 minutes instead of being re-tried on every call. A 429 gets a 30-second backoff, so a
  saturated bucket is stepped over rather than paid for on every turn.
- Cost: fill `PRICES` in `tracing.py` from the provider pricing pages.
- Everything shares a `turn_id`, passed through `RunnableConfig` into sub-agents, so one user
  message rolls up across the orchestrator and every agent it called.
- For the full trace tree (prompts, tool args, latency per step) add the LangSmith variables from
  `.env.example`. That sends prompts and completions to LangSmith; fine with the fake seed data.

## Orders and money
`OrderDetails.Amount` stores what the customer was charged, and `ListPrice` what the pair cost at
list, both copied in the same transaction that takes the stock. Reports read `o.Amount`, never
`s.Price`, so repricing a shoe cannot rewrite what past orders cost, and `ListPrice - Amount` is the
discount. `QuoteID` links a negotiated order to the quote it came from.

Payment lives in its own table. `Payment` holds one row per STK push - phone, amount in shillings,
`CheckoutRequestID`, and how it ended - and `OrderDetails.PaymentStatus` follows it from `UNPAID` to
`PENDING` to `PAID`. Only `tools.settle` writes `PAID`, and only after Daraja's own query API says so
and the amount matches the row, so a forged callback changes nothing. One pending prompt per order,
however many times a button is pressed.

Note `cancel_order` does not yet consider payment: cancelling a paid order would need a refund.

## Configuration
Beyond the API keys, all optional:

| Variable | Default | What it does |
|---|---|---|
| `MISTRAL_MODEL` | `ministral-8b-latest` | Mistral model. Run `check_models.py`; on the free tier most listed models 403 or 429. |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Groq model for the terminal agents. |
| `VOICE_MODEL_ORDER` | `groq:openai/gpt-oss-20b,groq:openai/gpt-oss-120b,mistral` | Voice model chain, fastest first. Each entry is its own rate-limit bucket. |
| `GROQ_REASONING_EFFORT` | `low` | gpt-oss emits reasoning before its first word; `low` is the floor. |
| `VOICE_MAX_TOKENS` | `160` | Reply cap. Spoken replies are one or two sentences. |
| `VOICE_TIMEOUT_S` / `LLM_TIMEOUT_S` | `8` / `20` | Per-call timeout. The SDK defaults are 60-120s, which freezes a turn. |
| `MODEL_BREAKER_TTL_S` | `600` | How long a dead model (401/403/404) is skipped. |
| `MODEL_RATE_LIMIT_TTL_S` | `30` | How long a rate-limited model (429) is skipped. |
| `MAX_HISTORY_TOKENS` | `0` (off) | Trim conversation history to roughly this many tokens. |
| `TRACE_PATH` | `traces.jsonl` | Where traces go; empty switches the file off. |
| `VOICE_SHARED_SECRET` | *(unset)* | Bearer token the voice endpoint requires. Set it before exposing the port. |
| `VOICE_PORT` | `8013` | Port for `voice_server.py`. |
| `VOICE_RECURSION_LIMIT` | `8` | Graph steps per voice turn: about four tool calls, so a confused model cannot stall a call. |
| `SERVER_PORT` | `8013` | Port for `server.py` - the website, WhatsApp and the M-Pesa callback. |
| `WEB_SECRET` | *(random per run)* | Signs the browser session cookie and its CSRF token. Set it, or everyone is signed out on restart. |
| `PUBLIC_URL` | *(unset)* | The tunnel URL, used for the links WhatsApp sends. |
| `KES_PER_USD` | `130` | Dollars to whole shillings, applied once at payment. |
| `MPESA_*` | *(unset)* | Daraja keys, till, passkey and callback token. Without them the shop works but cannot take payment. |
| `WA_*` | *(unset)* | WhatsApp Cloud API token, phone number id, app secret and verify token. Without them the channel is simply off. |

## Tests
Each runs against its own throwaway database under `tests/_tmp`:
```bash
python tests/test_negotiation.py   # the floor, consent, idempotency, and the bug above, replayed
python tests/test_schema.py        # migrations: fresh and upgraded databases end up identical
python tests/test_tracing.py       # fake models: turn rollups, fallbacks, denials, tool errors
python tests/test_voice_server.py  # the endpoint over SSE: protocol, sessions, filler, hang-up, sanitiser
python tests/test_agent_core.py    # circuit breaker, approval denials, history window, terminal tools
python tests/test_call.py          # the terminal call with a fake mic, voice and keyboard: streaming,
                                   #   cut-in, pauses, echo, nudges, typing, the endpointer on hard audio
python tests/test_mpesa.py         # Daraja: phone formats, the password recipe, every failure code named
python tests/test_shop.py          # self-signup and payment through the tools: guards, forged callbacks
python tests/test_web.py           # the website driven like a browser: prices, ownership, CSRF, SSE chat
python tests/test_whatsapp.py      # signed webhooks, voice notes, the Pay button, redelivered messages
python tests/smoke_voice.py        # LIVE: a real haggle against Groq. Spends free-tier quota.
python tests/smoke_voice_http.py   # LIVE: the same over real HTTP; start voice_server.py first.
python tests/smoke_call.py         # LIVE: a whole terminal call over real audio devices with nobody at
                                   #   the keyboard - needs VB-Audio Virtual Cable; silent on your speakers
python tests/smoke_call_cli.py     # LIVE: the same, but the real `python call.py` command, as its own process
```
On Windows, set `PYTHONIOENCODING=utf-8` if a test's own output fails to print.

## Real email
Turn on 2-Step Verification for your Google account, create an App Password,
and put your address and that password in `SMTP_USER` / `SMTP_PASSWORD`.

## Notes
- **Free-tier rate limits are the real constraint.** Groq's free tier is 8,000 tokens/minute per model
  and one voice turn costs ~3,800, so about two turns a minute per bucket. Stacking models in
  `VOICE_MODEL_ORDER` multiplies that; a paid Groq tier removes it.
- **Antivirus HTTPS scanning** (Avast, AVG, Kaspersky, ESET) re-signs every connection with its own
  root certificate, which Python's default `certifi` bundle does not trust, so every provider call
  fails with `CERTIFICATE_VERIFY_FAILED`. `truststore` makes Python use the Windows certificate store
  instead; it is in `requirements.txt` and injected in `agent_core.py`.
- If a model ID stops working, run `check_models.py` and copy a current one into `.env`.
- Mistral's free tier trains on your prompts, so use only the fake seed data here.
