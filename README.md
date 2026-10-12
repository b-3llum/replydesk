# ReplyDesk

ReplyDesk pre-writes Google review replies for small businesses in Philadelphia and Abington, PA.
We scrape a business's newest reviews, write a reply to every unanswered one, and text the owner a
private preview page: five replies they can copy right away, the rest unlocked for $29.

**88% unanswered.** In our first scrape (10 restaurants on Baltimore Avenue, West Philadelphia),
517 of 584 recent Google reviews had no reply from the owner. Across all four searches so far
(39 businesses, 2,206 reviews) it is 81%.

## Architecture

1. `scrape.py`: Apify. The Google Maps Scraper finds places (name, phone, rating); the Google Maps Reviews Scraper pulls the 60 newest reviews per place.
2. `store.py`: businesses, reviews (with the Studio's tags), orders and requests (home-page leads), in MongoDB Atlas or a JSON file behind the same functions. Config picks the backend (see Environment); `run.py migrate` copies the JSON store into Mongo.
3. `llm.py` + `prompts.py`: Luminal Cloud (OpenAI-compatible, DeepSeek-V4.1-Flash) with thinking off, throttled to the key's rate limit; system prompt, 3 few-shot pairs and a no-invented-facts guard. Every call is logged to `data/llm.log`.
4. `generate.py`: 40 threads; rule checks with one targeted rewrite; per-business pass for unique openings and closings and varied call lines; templates for rating-only reviews.
5. `app.py` + `templates/` + `static/`: FastAPI + Jinja, the pages below.
6. `studio.py` + `quality.py`: the Reply Studio: review list, LLM tags, the tone composer, and a 0-100 reply quality score from fixed rules (no LLM call).
7. `outreach.py`: who to text first, the outreach text and the CSV export.
8. `evolution.py`: the admin's prompt-evolution card. `evolve/` is the prompt-evolution experiment it reads; see `evolve/README.md`.
9. `config.py`: settings lookup (see Environment). `run.py`: the commands below. `smoke.py`: smoke checks.

Pages:

| Page | What it is |
|---|---|
| `/` | Landing page: live stats, Find my business (`POST /find`), a form to ask for a preview (`POST /request`, saved to requests) and the $19/month waitlist |
| `/p/<slug>?t=<token>` | Owner preview: 5 replies readable, the rest locked (their text never reaches the browser). A paid business redirects to `/full`. Back from `/thanks`, the page polls `status.json` and switches to the full set once the payment is marked |
| `/full/<slug>?t=<token>` | Every reply, once the business has paid; otherwise redirects to `/p` |
| `/thanks` | Where Stripe sends the owner after paying; links back to their replies |
| `/studio/<slug>?t=<token>` | Reply Studio (desktop): the review queue, LLM topic tags (cached on the review), a tone composer whose Regenerate and Shorten results are never saved, and the quality score |
| `/admin` | Sign in with `ADMIN_KEY` (the browser stays signed in for 7 days). Businesses: mark texted or paid, CSV export. Reviews: latest replies, Luminal usage and the prompt-evolution card, which needs `evolve/results.json` (git-ignored, so only where evolve ran). Orders. Outreach: texts to send and the home-page requests |
| `/health` | Store check: `{"ok": true, "store": ..., "db_ms": ...}`, or 503 when the store is down |

## Commands

Setup: Python 3.11 or newer (3.13 in use). `py -3.13 -m venv .venv`, then
`.venv\Scripts\pip install -r requirements.txt`, then copy `.env.example` to `.env` and fill it in
(or set the values with `setx`).

```
.venv\Scripts\python run.py scrape --query "restaurants Baltimore Avenue Philadelphia" --max 10 [--from-cache]
.venv\Scripts\python run.py generate --all          (or --business <slug> [--samples N], add --redo to rewrite)
.venv\Scripts\python run.py serve [--reload]        (0.0.0.0:8000)
.venv\Scripts\python run.py outreach                (data/outreach.csv, most unanswered first)
.venv\Scripts\python smoke.py [store] [llm] [web]   (every section when none is named)
```

- `--from-cache` reuses the raw Apify results in `data/raw/` (no Apify calls).
- `--samples N`: with `--business`, print N replies next to their reviews afterwards (default 5).
- `--reload` restarts the server when the code changes (for development).
- Also: `run.py list` (businesses and slugs), `run.py samples --business <slug> [--n N]`, and
  `run.py migrate` (copies `data/store.json` into Mongo; the JSON file stays).

Smoke sections (PASS/FAIL/SKIP per check, exit code 1 on any FAIL):

- `store`: `store.py` on temp JSON files and on throwaway `replydesk_smoke_*` databases on the configured cluster, dropped afterwards.
- `llm`: the prompt and template rules offline, then one live Luminal call.
- `web`: starts the app on a free local port and makes read-only requests against the configured store. No LLM calls.

## Running it

`run.py serve` listens on 0.0.0.0:8000. Put it behind a public HTTPS URL and set that URL as
`PUBLIC_BASE_URL`: preview links and outreach texts are built from it. In Stripe, send buyers to
`<PUBLIC_BASE_URL>/thanks` after payment; the payment's `client_reference_id` is the business slug.

There is no Stripe webhook: payments are marked by hand. When one comes in, open the admin and press
Mark paid on that business (Businesses page). Its preview link then opens the full set.

## Environment

Read from `.env` first, then the environment, then the Windows user registry (values set with `setx`).
Empty values are skipped, so an empty line in `.env` falls through to the next source.

| Variable | Needed for | Notes |
|---|---|---|
| `LUMINAL_BASE_URL` | generate, Reply Studio tags and composer, `smoke.py llm` | `https://flash.luminal.cloud/v1` |
| `LUMINAL_API_KEY` | same | Luminal key |
| `LUMINAL_MODEL` | same | `deepseek-ai/DeepSeek-V4.1-Flash` |
| `APIFY_TOKEN` | scrape (not with `--from-cache`) | Apify API token |
| `MONGODB_URI` | the Mongo store | Atlas connection string; empty means the JSON store |
| `PUBLIC_BASE_URL` | preview links, outreach texts | Public HTTPS URL of the app. The admin Businesses and Outreach pages, the CSV export and `run.py outreach` fail without it |
| `PAYMENT_LINK` | the Unlock buttons | Stripe payment link; we add `client_reference_id=<slug>`. Empty: the buttons go nowhere |
| `ADMIN_KEY` | `/admin` | Admin sign-in key. Empty: the admin stays locked (unless demo mode is on) |
| `DEMO_MODE` | demo mode | `1`, `true` or `yes`. Off unless set (see Demo mode) |
| `REPLYDESK_STORE` | picking the store | `json` or `mongo`; empty (or `auto`) means mongo when `MONGODB_URI` is set, else json |
| `REPLYDESK_JSON_PATH` | the JSON store | Default `data/store.json`, relative to the app folder |
| `REPLYDESK_MONGO_DB` | the Mongo store | Database name, default `replydesk` |

`.env`, `DEMO_MODE` and the store settings are read once at startup: restart the server after
changing them. `evolve/` has its own settings; see `evolve/README.md`.

## Demo mode

Set `DEMO_MODE=1` in `.env` and restart `python run.py serve`. The home page then shows "Take the
tour" (an owner preview, a paid full set, the Reply Studio and the admin), and the admin opens
without a key.

Demo mode publishes live preview links and leaves the admin, including Mark paid and the CSV export,
open to anyone. Use it only on a local or private server, never on the public URL. It is off unless
set; to override a value set with `setx`, put `DEMO_MODE=0` in `.env` (an empty value falls through).
