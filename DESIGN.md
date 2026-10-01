# Mo's Shoe Shop — the visual world

The shop is built out of **kanga**: the printed cotton every market in Nairobi is wrapped in. A kanga
is three things — a border print (*pindo*), a saturated field, and a proverb along the hem — and so
is this shop.

Two rules decide most arguments:

1. **The pattern lives in the borders.** The middle of the page is for goods and prices, never for
   decoration.
2. **The number is the hero.** A price is what the customer came to argue with, so on every page the
   largest, boldest thing is a number: the tag, the total, the shilling amount on a receipt.

## Colour

| Token | Value | What it is |
|---|---|---|
| `--night` | `#15235f` | The indigo a kanga is dyed. The ground of every page. |
| `--cloth` | `#f4ecdd` | Unbleached cotton. What the shop writes on indigo with. |
| `--paper` | `#fbf6ec` | A label, a slip, a receipt: paper pinned to the cloth. |
| `--ink` | `#17120d` | Warm printing ink, for anything written on paper. The header bar. |
| `--mustard` | `#edaa14` | The third dye. **Price tags only**, plus the hero call to action. |
| `--flame` | `#d2372e` | The red of a border print. A fill and a hover state, never small text — it does not carry on indigo. |
| `--mpesa` / `--mpesa-deep` | `#00a651` / `#00663a` | Money keeps its own green, and only money gets it. |
| `--faint` / `--quiet` | `#c7bfab` / `#5c5248` | Small print, on indigo and on paper respectively. |

Buttons are cloth with ink text, because a mustard button beside a mustard tag reads as one object.

## Type

- **Bricolage Grotesque** 800, tracking −0.035em: headings and every price numeral. Variable `opsz`,
  so the large sizes tighten up by themselves.
- **Karla** 400/500/700: everything that is read rather than shouted.
- Prices and amounts are `tabular-nums` wherever they sit in a column.

## Layout

- A `pindo` band (an SVG motif of mustard and flame diamonds on indigo) under the header and above
  the footer. Nowhere else.
- One column: 34rem on a phone, 46rem from 40rem up, 52rem from 64rem up. Prose keeps a 62ch measure
  whatever the column does.
- The stall is rows with hairline rules, not cards. Each row is a name, a description, a price tag
  cut from mustard card with a punched hole, then Buy and Make an offer.
- An order is a paper slip on the cloth, with a dashed tear line above the total.
- The chat is Mo on paper, the customer on a mustard slip, and a fixed ask bar in ink.

## Motion

One authored moment: the **stamp** when an order turns PAID, scaling and straightening out of a
tilt, ease-out-expo. Everything else is a 120ms colour change on hover. `prefers-reduced-motion`
removes both.

## Voice

Plain, warm, and in the shop's own words. A Swahili proverb heads each page with its English on the
same line — *Bei ni mazungumzo*, a price is a conversation. Buttons name the action they perform
("Make an offer", "Pay KSh 11,699 by M-Pesa"), empty states say what to do next, and errors say what
happened rather than apologising.
