# ReplyDesk

ReplyDesk pre-writes Google review replies for small businesses in Philadelphia and Abington, PA.
We scrape a business's newest reviews, write a reply to every unanswered one, and text the owner a
private preview page: five replies they can copy right away, the rest unlocked for $29.

**88% unanswered.** In our first scrape (10 restaurants on Baltimore Avenue, West Philadelphia),
517 of 584 recent Google reviews had no reply from the owner. Across all four searches so far
(39 businesses, 2,206 reviews) it is 81%.

## Architecture

1. `scrape.py`: Apify. The Google Maps Scraper finds places (name, phone, rating); the Google Maps Reviews Scraper pulls the 60 newest reviews per place.
2. `store.py`: MongoDB Atlas (businesses, reviews, orders), with a JSON-file fallback behind the same functions; `run.py migrate` moves JSON into Mongo.
3. `llm.py` + `prompts.py`: Luminal Cloud (OpenAI-compatible, DeepSeek-V4.1-Flash) with thinking off, throttled to the key's rate limit; system prompt, 3 few-shot pairs and a no-invented-facts guard.
4. `generate.py`: 40 threads; rule checks with one targeted rewrite; per-business pass for unique openings and closings and varied call lines; templates for rating-only reviews.
5. `app.py`: FastAPI + Jinja. Owner preview (locked replies never reach the browser), paid full set, Stripe thanks page, admin dashboard with live stats.
6. `outreach.py` + `evolution.py`: outreach CSV and texts; the admin's prompt-evolution card reads `evolve/results.json`.

## Commands

Setup: `py -3.13 -m venv .venv` then `.venv\Scripts\pip install -r requirements.txt`.

```
.venv\Scripts\python run.py scrape --query "restaurants Baltimore Avenue Philadelphia" --max 10
.venv\Scripts\python run.py generate --all          (or --business <slug>, add --redo to rewrite)
.venv\Scripts\python run.py serve                   (0.0.0.0:8000)
.venv\Scripts\python run.py outreach                (data/outreach.csv, most unanswered first)
```

Also: `run.py list` (businesses and slugs), `run.py samples --business <slug>`, `run.py migrate`,
and `python smoke.py` (store and LLM checks).

## Environment

Read from `.env` first, then the environment, then the Windows user registry (values set with `setx`).

| Variable | What it is |
|---|---|
| `LUMINAL_BASE_URL` | `https://flash.luminal.cloud/v1` |
| `LUMINAL_API_KEY` | Luminal key |
| `LUMINAL_MODEL` | `deepseek-ai/DeepSeek-V4.1-Flash` |
| `APIFY_TOKEN` | Apify API token |
| `MONGODB_URI` | Atlas connection string; empty means the JSON store in `data/` |
| `PUBLIC_BASE_URL` | Public URL of the app, used in preview links |
| `PAYMENT_LINK` | Stripe payment link (we add `client_reference_id=<slug>`) |
| `ADMIN_KEY` | Key for `/admin` |
| `REPLYDESK_STORE` | Optional: `json` or `mongo` to force a store |

## Demo

- Admin: https://bellum.ngrok.app/admin?key=ADMIN_KEY (use the real key from your environment)
- Owner preview (Loco Pez West Philly): https://bellum.ngrok.app/p/loco-pez-west-philly-b71a?t=xHQir8h2fjE
- Health check: https://bellum.ngrok.app/health
