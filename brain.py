"""One turn of Mo's, for any channel: the web, WhatsApp, a phone call, the terminal.

voice_server.py already runs turns for ElevenLabs, but in that protocol's shape (OpenAI SSE, a
transcript resent every turn, end_call). The web and WhatsApp need the same brain without any of
that, so this is the plain version:

    brain = Brain(voice_server.build_server_agent())
    async for kind, said in brain.turn("wa:254712345678", "how much is the trail runner?"):
        ...

Same agent, same tools, same haggling and consent guards, same traces. What a channel adds is who
it is talking to: `known` becomes a note the model sees, and the phone number the channel has
verified reaches the tools as `channel_phone`, where create_customer trusts it over anything the
model types.

Sentences come out one at a time, sanitised for speech, because a sentence is what both
text-to-speech and a chat bubble want.
"""
import asyncio

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from tracing import TRACER
from voice_server import (FALLBACK_REPLY, FILLER, RECURSION_LIMIT, _cut, _drop_repeats, _text,
                          speakable)


class Brain:
    """The shop's one conversation engine. Safe to share: turns are serialised per session."""

    def __init__(self, agent, recursion_limit: int = RECURSION_LIMIT):
        self.agent = agent
        self.recursion_limit = recursion_limit
        self.locks: dict[str, asyncio.Lock] = {}
        # TRACER keeps one turn open at a time, so two channels mid-turn would write into each
        # other's rollup. One lock across all sessions costs nothing on a free tier.
        self.tracing = asyncio.Lock()

    def lock(self, session: str) -> asyncio.Lock:
        return self.locks.setdefault(session, asyncio.Lock())

    async def turn(self, session: str, text: str, known: dict | None = None, channel: str = "web"):
        """Yield ("say", sentence) as Mo says them, then ("done", meta) once.

        meta: {"ordered", "order_id", "turn_id", "error", "said"}.
        """
        said_text = (text or "").strip()
        if not said_text:
            yield "done", {"ordered": False, "order_id": None, "turn_id": "", "error": "nothing said",
                           "said": ""}
            return
        out: list[str] = []
        pending, spoken, repeats = "", [], set()

        def ready(piece: str, final: bool = False):
            """Hold the model's words until a whole sentence is there, then queue it to go out."""
            nonlocal pending
            pending += piece
            cut = len(pending) if final else _cut(pending)
            if not cut:
                return
            whole, pending = pending[:cut], pending[cut:]
            line = _drop_repeats(speakable(whole), repeats).strip()
            if line:
                spoken.append(line)
                out.append(line)

        async with self.lock(session), self.tracing:
            turn_id = TRACER.begin_turn(said_text)
            config = {"configurable": {
                "thread_id": session, "turn_id": turn_id, "depth": 0,
                # The caller's own words, clean: place_order and create_customer read these to tell a
                # yes from a question, so the channel note must never be mixed in.
                "utterance": said_text,
                "channel": channel,
                "channel_phone": (known or {}).get("phone", ""),
                # Who the channel has established this is, which is the only identity the tools
                # trust. It never comes from the model, so no amount of typing can change it.
                "customer_id": (known or {}).get("customer_id"),
            }, "recursion_limit": self.recursion_limit}
            ordered, order_id, error = False, None, None
            try:
                async for message, where in self.agent.astream(
                        {"messages": [HumanMessage(self._with_note(said_text, known))]}, config,
                        stream_mode="messages"):
                    if where.get("langgraph_node") != "agent":
                        continue
                    piece = _text(message.content)
                    if piece:
                        ready(piece)
                    for call in (getattr(message, "tool_call_chunks", None)
                                 or getattr(message, "tool_calls", None) or []):
                        name = call.get("name")
                        if name == "place_order":
                            ordered = True
                        # A tool takes a second or two. Saying something first is the difference
                        # between a shop assistant looking something up and a silent screen.
                        if name in FILLER and not spoken:
                            ready(FILLER[name] + " ", final=True)
                    while out:
                        yield "say", out.pop(0)
                ready("", final=True)
            except GraphRecursionError:
                error = "recursion limit"
                ready(FALLBACK_REPLY, final=True)
            except Exception as e:                       # every model down, or a bug
                error = f"{type(e).__name__}: {e}"
                ready(FALLBACK_REPLY, final=True)
            finally:
                # A browser that closes the tab cancels this generator mid-yield. Without a finally
                # the turn would stay open in the tracer and the next one would write into it.
                TRACER.end_turn(error)
            while out:
                yield "say", out.pop(0)
            if ordered:
                order_id = self._newest_order(known)
        yield "done", {"ordered": ordered, "order_id": order_id, "turn_id": turn_id, "error": error,
                       "said": " ".join(spoken)}

    @staticmethod
    def _with_note(text: str, known: dict | None) -> str:
        """What the model sees. The note says who it is talking to, so a returning customer is not
        asked their name again and a new one is not asked for a number the channel already has."""
        bits = []
        if (known or {}).get("customer_id"):
            bits.append(f"customer_id={known['customer_id']} ({known.get('name') or 'on file'})")
        else:
            # Nobody has said who they are. Looking them up anyway turns "how much is the trail
            # runner?" into "I can't find that name", which is what a live visitor got.
            bits.append("you do not know who this is and they have not given a name; do not look a "
                        "customer up - talk about the shoes, and ask for a name and an M-Pesa "
                        "number only when they want to order")
        if (known or {}).get("phone"):
            bits.append(f"messaging from {known['phone']}")
        return f"(shop note, not spoken aloud: {'; '.join(bits)})\n{text}"

    @staticmethod
    def _newest_order(known: dict | None) -> int | None:
        """The order this turn placed, so the channel can offer to take payment for it."""
        from db import db

        # Only ever this customer's own order. Without a customer there is nobody to answer for, and
        # the newest order in the shop belongs to somebody else.
        if not (known and known.get("customer_id")):
            return None
        with db() as conn:
            row = conn.execute("SELECT OrderID FROM OrderDetails WHERE CustomerID=? "
                               "ORDER BY OrderID DESC LIMIT 1", (known["customer_id"],)).fetchone()
        return row["OrderID"] if row else None


def customer_by_phone(phone: str) -> dict:
    """Who is this number, if anyone? What a channel passes to Brain.turn as `known`."""
    import mpesa
    from db import db

    try:
        number = mpesa.phone_number(phone)
    except ValueError:
        return {}
    with db() as conn:
        # Not a web profile: anybody can type anybody's number into a form, so a web record never
        # speaks for a number on another channel. WhatsApp's own verification does.
        row = conn.execute("SELECT CustomerID, CustomerName FROM CustomerInfo WHERE Phone=? "
                           "AND Source <> 'web' ORDER BY CustomerID LIMIT 1", (number,)).fetchone()
    known = {"phone": number}
    if row:
        known.update(customer_id=row["CustomerID"], name=row["CustomerName"])
    return known
