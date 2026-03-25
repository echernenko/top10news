# 🗞️ Top 10 News Agent

A zero-cost news agent that curates your daily top 10 stories, published as a static page on GitHub Pages.

| Section | Stories | Sources |
|---------|---------|---------|
| 🌍 World | 6 | BBC, NPR, Al Jazeera, AP, Reuters, The Guardian |
| 🇺🇦 Ukraine | 2 | Ukrainska Pravda, Ukrinform, Kyiv Independent, Babel |
| 🌲 Washington State | 2 | Seattle Times, KING5, Crosscut, KUOW, Seattle PI |

## How it works

1. **GitHub Actions** runs every 6 hours (or on manual trigger)
2. **RSS feeds + web scraping** pull headlines from credible sources
3. Headlines are sent to an **LLM via GitHub Models** (free) to pick the top stories per category
4. A static **HTML page + JSON file** are generated and deployed to **GitHub Pages**

## Setup

```bash
# Clone / init
git init top10news && cd top10news
# Copy generate.py, .github/workflows/update-news.yml, .gitignore

git add -A && git commit -m "init"
gh repo create top10news --public --push --source=.
```

Then go to **Settings → Pages → Source → GitHub Actions**, and trigger the first run:

```bash
gh workflow run "Update News"
```

Your page will be live at `https://<your-username>.github.io/top10news/`

## Customization

- **Update frequency** – edit the cron in `.github/workflows/update-news.yml`
- **Sources** – edit `SOURCES` dict in `generate.py` (add/remove RSS feeds or scrape targets)
- **Story counts** – edit the `(category, count)` tuples in `main()`
- **LLM model** – change `model` in `call_llm()` (see [GitHub Models](https://github.com/marketplace/models))
- **Look & feel** – edit the CSS in `render_html()`

## Cost

**$0.** GitHub Actions free tier + GitHub Models free tier. Each run takes ~60 seconds.

