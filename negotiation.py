"""The haggling policy. Deterministic, pure, and the only thing allowed to decide a price.

The split that makes this safe: the model does the language of bargaining - warmth, reluctance,
knowing when to hold - and this module does the arithmetic. The model is never told the floor,
so it cannot leak it, be argued past it, or split the difference on its own. Same instinct as
`WHERE ShoeID=? AND InvCount > 0` in place_order: do not ask the model to be disciplined about
a number when SQL can simply refuse.

Every price is a whole dollar, because "ninety five" reads aloud and "ninety five point nine
nine" does not, and because rounding up can never breach a floor.
"""
import math
import re

QUOTE_TTL_MINUTES = 10
# Share of the remaining gap conceded on each round: generous, then stingy, then nearly nothing.
CONCESSION = (0.50, 0.30, 0.15)
ACCEPT_WITHIN = 0.03   # close enough to the ask that haggling further would be petty
MAX_ROUNDS = 3         # after this the answer stops moving, whatever the customer says
# An offer under half the floor is not a serious offer, and must not buy a concession: otherwise
# offering one dollar three times lands on the floor faster than any honest offer, and anyone
# who tries it learns exactly where the floor is.
LOWBALL = 0.5

ACCEPT, COUNTER, FINAL, HOLD = "ACCEPT", "COUNTER", "FINAL", "HOLD"


def _up(x: float) -> int:
    """Round up. Never down: rounding must not be a way through the floor."""
    return int(math.ceil(x - 1e-9))


def _share(round_n: int) -> float:
    return CONCESSION[min(max(round_n, 1) - 1, len(CONCESSION) - 1)]


def decide(asking: float, floor: float, customer_offer: float, round_n: int = 1) -> tuple[str, int]:
    """Return (verdict, price) for a customer's offer. Price is always a whole dollar.

    `asking` is the last price the shop named (list price on round one). `floor` is the lowest
    acceptable price; 0 or anything at or above the ask means the pair is simply not negotiable.
    """
    ask = _up(asking)
    if floor <= 0 or floor >= asking:
        return FINAL, ask                       # not negotiable: one price, stated once
    low = _up(floor)
    # The offer arrives as a JSON number the model produced: refuse to reason about NaN, infinity
    # or a negative, rather than letting them fall through comparisons in surprising ways.
    if not math.isfinite(customer_offer) or customer_offer < low * LOWBALL:
        return HOLD, ask                        # not a serious offer: the price does not move

    if customer_offer >= ask:
        return ACCEPT, ask                      # never charge more than we last asked

    if customer_offer >= low:
        if customer_offer >= ask * (1 - ACCEPT_WITHIN) or round_n >= MAX_ROUNDS:
            # Take their number, but never below the floor after rounding.
            return ACCEPT, max(low, int(customer_offer))
        return COUNTER, max(low, _up(ask - (ask - customer_offer) * _share(round_n)))

    # Below the floor. Concede toward it a share at a time, and when the haggling is over say
    # the real best price rather than wherever the concession schedule happened to stop - a
    # seller who says "that's my final offer" and is still holding margin is not final.
    if round_n >= MAX_ROUNDS:
        return FINAL, low
    return COUNTER, max(low, _up(ask - (ask - low) * _share(round_n)))


# ---------- prices the customer names out loud ----------
# Consent needs more than "the customer said something after the quote": "that's too steep, I'll
# give you eighty" is a reply, and it is a refusal. place_order uses this to notice a counter-offer
# in the very utterance the model is treating as a yes. Only numbers in a money context count, so
# a shoe size ("size 42") or a quantity ("two pairs") is never mistaken for an offer.
_UNITS = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
          "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_NUMBER_WORDS = set(_UNITS) | set(_TENS) | {"hundred", "and", "a"}
_BEFORE = {"give", "pay", "offer", "offering", "take", "do", "about", "for", "at"}
_CURRENCY = {"dollar", "dollars", "bucks", "bob", "shillings", "shilling", "ksh", "kes"}
_NOT_MONEY_BEFORE = {"size", "sizes", "number", "no"}
_NOT_MONEY_AFTER = {"pair", "pairs", "size", "sizes", "percent"}
_TOKEN = re.compile(r"\$?\d+(?:\.\d+)?|[a-z]+")


def _words_value(words: list[str]) -> int | None:
    """'eighty five' -> 85, 'a hundred and eight' -> 108, 'one twenty' -> 120, 'one oh eight' -> 108."""
    words = [w for w in words if w not in ("a", "and")]
    if not words or not all(w in _UNITS or w in _TENS or w == "hundred" for w in words):
        return None
    # Prices are said colloquially: "one twenty" is 120, not 21. A leading single digit followed
    # by more number words, with no "hundred", is hundreds.
    if len(words) >= 2 and "hundred" not in words and _UNITS.get(words[0], 99) < 10:
        rest = _words_value(words[1:])
        return None if rest is None else _UNITS[words[0]] * 100 + rest
    total = 0
    for w in words:
        if w == "hundred":
            total = (total or 1) * 100
        else:
            total += _UNITS.get(w, 0) + _TENS.get(w, 0)
    return total


def spoken_offers(text: str, money_only: bool = True, bare: bool = False) -> list[float]:
    """Every amount of money the customer names in `text`, in order.

    money_only=True counts only numbers in a money context, so "size 42" is never an offer.
    money_only=False (for allowing an offer) counts every number said: after "what price were you
    thinking?", a bare "Eighty." is an offer. bare=True (for refusing an order) also counts a number
    said on its own - "Okay, ninety." to a quote of 108 is a counter-offer, not a yes - but still
    never a size, a number of pairs or a percentage.
    """
    tokens = _TOKEN.findall((text or "").lower().replace("-", " ").replace(",", ""))
    found, i, last_money_end = [], 0, -1
    while i < len(tokens):
        tok, j = tokens[i], i + 1
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if tok.lstrip("$")[:1].isdigit():
            value, dollar_sign = float(tok.lstrip("$")), tok.startswith("$")
        elif tok in _UNITS or tok in _TENS or tok == "hundred" or (tok == "a" and nxt == "hundred"):
            while j < len(tokens) and tokens[j] in _NUMBER_WORDS:
                j += 1
            while j > i + 1 and tokens[j - 1] in ("a", "and"):  # "eighty and" -> "eighty"
                j -= 1
            value, dollar_sign = _words_value(tokens[i:j]), False
            if value is None:
                i += 1
                continue
        else:
            i += 1
            continue
        before, after = tokens[max(0, i - 2):i], tokens[j] if j < len(tokens) else ""
        excluded = (before and before[-1] in _NOT_MONEY_BEFORE) or after in _NOT_MONEY_AFTER
        # A number straight after another offer is an offer too: "would you take 80? 85? 90?"
        money = (dollar_sign or after in _CURRENCY or any(b in _BEFORE for b in before)
                 or i == last_money_end)
        if not money_only or ((money or bare) and not excluded):
            found.append(float(value))
            last_money_end = j
        i = j
    return found


# ---------- a yes you can hear ----------
# On a phone line the caller's words arrive through speech-to-text. A live test call turned "That's a
# bit steep. I'll give you eighty dollars." into "Deep.", and the model ordered on it at full price.
# So the words the model treats as agreement must contain one. A doubtful yes costs one more question
# ("shall I put that through?"); a false yes costs a customer an order they never agreed to.
_YES = re.compile(
    r"\b(yes|yeah|yep|yup|ya|sure|ok|okay|alright|all right|deal|fine|agreed|agree|done|sold|"
    r"perfect|great|good|go ahead|go on|go for it|do it|sounds good|sounds great|works|"
    r"take it|take them|take that|take those|have it|have them|buy it|buy them|"
    r"place it|place the order|order it|order them|book it|put it through|put that through|"
    r"ring it up|confirm|correct|absolutely|definitely|of course|please do|let'?s go|why not|"
    r"that'?s right|(?:that|then|\d+) it is)\b", re.I)
# ...unless the same words hold back: "no, that's fine", "okay, that's too much", "not sure". Nor is a
# question an answer - "Okay, so what sizes do you have?" opens with a yes-word and agrees to nothing -
# and nor is "yes, I'm here", which answers "are you still there?", not "shall I put it through?".
_HOLD_BACK = re.compile(
    r"^\W*(no|nope|nah)\b|\bno deal\b|\?|\b(not|don'?t|do not|never|cancel|wait|hold on|hang on|"
    r"maybe|later|think about it|let me think|too much|too expensive|too steep|too high|too pricey|"
    r"steep|expensive|pricey|lowest|lower|less|cheaper|discount|better price|best price|"
    r"what|which|how|when|where|who|why|is it|is that|are they|do you|does it|can you|could you|"
    r"still here|still there|i'?m here|i am here|hear me|hello)\b", re.I)
# A question that asks for the sale is a yes all the same: "Can you put it through?"
_ASKS_TO_BUY = re.compile(r"\b(can|could|would|will) you (please )?(put|place|ring|book|order|wrap)\b", re.I)


def said_yes(text: str) -> bool:
    """Do these words agree to buy? Plain agreement, and nothing in them that holds back."""
    text = re.sub(r"\bwhy not\b\W*", "yes ", text or "", flags=re.I)
    if not _YES.search(text):
        return False
    rest = _ASKS_TO_BUY.sub(" ", text)
    if rest != text:
        rest = rest.replace("?", " ")      # the question WAS the request
    return not _HOLD_BACK.search(rest)
