"""
Top 10 News Agent
- 2 from Washington State
- 2 from Ukraine
- 6 from the World
Fetches RSS feeds + scrapes websites, curates via LLM (GitHub Models).
"""

import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html.parser import HTMLParser

# ── Sources by category ──────────────────────────────────────────────────────

SOURCES = {
    "wa": {
        "feeds": [
            ("The Seattle Times", "https://www.seattletimes.com/feed/"),
            ("KING5 News", "https://www.king5.com/feeds/syndication/rss/news"),
            ("Crosscut", "https://crosscut.com/feed"),
        ],
        "scrape": [
            ("KUOW", "https://www.kuow.org/stories"),
            ("Seattle PI", "https://www.seattlepi.com/local/"),
        ],
    },
    "ukraine": {
        "feeds": [
            ("Ukrainska Pravda", "https://www.pravda.com.ua/eng/rss/"),
            ("Ukrinform", "https://www.ukrinform.net/rss/block-lastnews"),
            ("Kyiv Independent", "https://kyivindependent.com/feed/"),
        ],
        "scrape": [
            ("Babel", "https://babel.ua/en"),
        ],
    },
    "world": {
        "feeds": [
            ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
            ("NPR News", "https://feeds.npr.org/1001/rss.xml"),
            ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
            ("Associated Press", "https://rsshub.app/apnews/topics/world-news"),
            ("Reuters", "https://rsshub.app/reuters/world"),
            ("The Guardian", "https://www.theguardian.com/world/rss"),
        ],
        "scrape": [],
    },
}


# ── Simple HTML text extractor ────────────────────────────────────────────────

class TextExtractor(HTMLParser):
    """Extracts visible text from HTML, focusing on headlines and article links."""

    def __init__(self):
        super().__init__()
        self._items = []
        self._current_text = ""
        self._current_href = ""
        self._in_headline = False
        self._headline_tags = {"h1", "h2", "h3", "h4"}
        self._skip_tags = {"script", "style", "nav", "footer", "aside", "form", "noscript"}
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return

        attrs_dict = dict(attrs)
        if tag in self._headline_tags:
            self._in_headline = True
            self._current_text = ""
            self._current_href = ""
        if tag == "a" and self._in_headline:
            self._current_href = attrs_dict.get("href", "")

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth > 0:
            return

        if tag in self._headline_tags and self._in_headline:
            text = self._current_text.strip()
            if text and len(text) > 10:
                self._items.append({"title": text, "link": self._current_href})
            self._in_headline = False

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        if self._in_headline:
            self._current_text += data

    def get_items(self):
        return self._items


# ── Fetching ──────────────────────────────────────────────────────────────────

def fetch_feed(name: str, url: str) -> list[dict]:
    """Parse an RSS feed and return headline dicts."""
    items = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NewsAgent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            tree = ET.parse(resp)
        for item in tree.iter("item"):
            title = item.findtext("title", "").strip()
            link = item.findtext("link", "").strip()
            desc = item.findtext("description", "").strip()
            desc = re.sub(r"<[^>]+>", "", desc).strip()
            if title and len(title) > 5:
                items.append(
                    {"source": name, "title": title, "link": link, "description": desc[:300]}
                )
    except Exception as e:
        print(f"    ⚠ Feed failed {name}: {e}")
    return items


def scrape_headlines(name: str, url: str) -> list[dict]:
    """Scrape headlines from a web page by extracting h1-h4 text."""
    items = []
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; NewsAgent/1.0)",
                "Accept": "text/html",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        parser = TextExtractor()
        parser.feed(html)

        seen = set()
        for item in parser.get_items():
            title = item["title"]
            if title not in seen:
                seen.add(title)
                link = item["link"]
                if link and not link.startswith("http"):
                    from urllib.parse import urljoin
                    link = urljoin(url, link)
                items.append(
                    {"source": name, "title": title, "link": link or url, "description": ""}
                )
    except Exception as e:
        print(f"    ⚠ Scrape failed {name}: {e}")
    return items


def fetch_category(category: str) -> list[dict]:
    """Fetch all headlines for a category (feeds + scraping)."""
    cfg = SOURCES[category]
    all_items = []

    for name, url in cfg["feeds"]:
        print(f"    RSS: {name}")
        all_items.extend(fetch_feed(name, url))

    for name, url in cfg["scrape"]:
        print(f"    Scrape: {name}")
        all_items.extend(scrape_headlines(name, url))

    print(f"    → {len(all_items)} headlines")
    return all_items[:40]


# ── LLM curation ─────────────────────────────────────────────────────────────

def call_llm(prompt: str, system: str = "You are a concise, neutral news editor.") -> str:
    """Call GitHub Models LLM and return raw content string."""
    api_key = os.environ.get("GITHUB_TOKEN", "")
    if not api_key:
        raise RuntimeError("GITHUB_TOKEN not set")

    body = json.dumps(
        {
            "model": "openai/gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 2000,
        }
    ).encode()

    req = urllib.request.Request(
        "https://models.inference.ai.azure.com/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    content = data["choices"][0]["message"]["content"]
    content = re.sub(r"^```json\s*", "", content.strip())
    content = re.sub(r"\s*```$", "", content.strip())
    return content


def curate_category(headlines: list[dict], category: str, count: int) -> list[dict]:
    """Use LLM to pick top N stories from a list of headlines."""
    label = {
        "wa": "Washington State (local WA news — Seattle, Puget Sound, WA politics, etc.)",
        "ukraine": "Ukraine (war, politics, society, reconstruction)",
        "world": "World (global affairs, excluding Ukraine and WA state)",
    }[category]

    headlines_text = "\n".join(
        f"- [{h['source']}] {h['title']}"
        + (f" | {h['description'][:120]}" if h["description"] else "")
        for h in headlines
    )

    prompt = f"""Below are recent headlines from {label} sources.

Pick the TOP {count} most important and newsworthy stories. For each, provide:
1. A clear headline (your own concise wording)
2. A 2-3 sentence summary
3. The original source name and URL from the list

Respond ONLY with valid JSON array, no wrapping object, no markdown fences:
[
  {{
    "headline": "...",
    "summary": "...",
    "source": "...",
    "url": "..."
  }}
]

Headlines:
{headlines_text}"""

    content = call_llm(prompt)
    stories = json.loads(content)
    return stories[:count]


# ── HTML rendering ────────────────────────────────────────────────────────────

SECTION_CONFIG = {
    "wa": {"label": "Washington State", "icon": "🌲", "accent": "#4a9e6d"},
    "ukraine": {"label": "Ukraine", "icon": "🇺🇦", "accent": "#f0c040"},
    "world": {"label": "World", "icon": "🌍", "accent": "#d4a853"},
}


def render_html(sections: dict[str, list[dict]]) -> str:
    """Render the full HTML page with sectioned news."""
    now = datetime.now(timezone.utc)
    time_str = now.strftime("%B %d, %Y at %H:%M UTC")

    sections_html = ""
    story_num = 0
    for cat in ["world", "ukraine", "wa"]:
        cfg = SECTION_CONFIG[cat]
        stories = sections.get(cat, [])

        stories_cards = ""
        for s in stories:
            story_num += 1
            stories_cards += f"""
            <article class="story" style="--section-accent: {cfg['accent']}">
                <span class="story-number">{story_num:02d}</span>
                <h2>{s['headline']}</h2>
                <p class="summary">{s['summary']}</p>
                <a href="{s['url']}" target="_blank" rel="noopener">
                    {s['source']} ↗
                </a>
            </article>"""

        sections_html += f"""
        <section class="news-section">
            <div class="section-header" style="--section-accent: {cfg['accent']}">
                <span class="section-icon">{cfg['icon']}</span>
                <h2 class="section-title">{cfg['label']}</h2>
                <span class="story-count">{len(stories)} stories</span>
            </div>
            {stories_cards}
        </section>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Top 10 News</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #0a0a0a;
            --fg: #e8e4df;
            --muted: #787068;
            --card-bg: #141210;
            --border: #2a2622;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            background: var(--bg);
            color: var(--fg);
            font-family: 'DM Sans', sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 3rem 1.5rem;
        }}

        header {{
            text-align: center;
            margin-bottom: 3rem;
        }}

        header h1 {{
            font-family: 'Instrument Serif', serif;
            font-size: clamp(2.5rem, 6vw, 4.5rem);
            font-weight: 400;
            letter-spacing: -0.02em;
            line-height: 1.1;
            color: var(--fg);
        }}

        header h1 span {{
            color: #d4a853;
        }}

        .subtitle {{
            color: var(--muted);
            font-size: 0.95rem;
            margin-top: 0.5rem;
        }}

        .timestamp {{
            color: var(--muted);
            font-size: 0.8rem;
            margin-top: 0.5rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}

        .content {{
            max-width: 680px;
            width: 100%;
        }}

        .news-section {{
            margin-bottom: 2.5rem;
        }}

        .section-header {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.25rem;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid var(--section-accent, var(--border));
        }}

        .section-icon {{
            font-size: 1.4rem;
        }}

        .section-title {{
            font-family: 'Instrument Serif', serif;
            font-size: 1.4rem;
            font-weight: 400;
            color: var(--section-accent, var(--fg));
            flex: 1;
        }}

        .story-count {{
            color: var(--muted);
            font-size: 0.75rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}

        .story {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.75rem;
            position: relative;
            transition: border-color 0.3s ease;
            margin-bottom: 1rem;
        }}

        .story:hover {{
            border-color: var(--section-accent, var(--border));
        }}

        .story-number {{
            font-family: 'Instrument Serif', serif;
            font-size: 2.5rem;
            color: var(--section-accent, #d4a853);
            opacity: 0.25;
            position: absolute;
            top: 0.75rem;
            right: 1.25rem;
            line-height: 1;
        }}

        .story h2 {{
            font-family: 'Instrument Serif', serif;
            font-size: 1.35rem;
            font-weight: 400;
            line-height: 1.3;
            margin-bottom: 0.6rem;
            padding-right: 2.5rem;
        }}

        .story .summary {{
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.6;
            margin-bottom: 0.75rem;
        }}

        .story a {{
            color: var(--section-accent, #d4a853);
            text-decoration: none;
            font-size: 0.8rem;
            font-weight: 500;
            letter-spacing: 0.03em;
            transition: opacity 0.2s;
        }}

        .story a:hover {{ opacity: 0.7; }}

        footer {{
            margin-top: 2rem;
            color: var(--muted);
            font-size: 0.75rem;
            text-align: center;
            line-height: 1.6;
        }}

        @media (max-width: 500px) {{
            body {{ padding: 2rem 1rem; }}
            .story {{ padding: 1.25rem; }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>Top <span>10</span> News</h1>
        <p class="subtitle">World · Ukraine · Washington State</p>
        <p class="timestamp">Updated {time_str}</p>
    </header>
    <div class="content">
        {sections_html}
    </div>
    <footer>
        Curated by AI from credible sources<br>
        Updated every 6 hours
    </footer>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("🗞️  Top 10 News Agent\n")

    sections = {}

    for category, count in [("world", 6), ("ukraine", 2), ("wa", 2)]:
        label = SECTION_CONFIG[category]["label"]
        print(f"\n── {label} ({count} stories) ──")

        print("  Fetching headlines...")
        headlines = fetch_category(category)

        if not headlines:
            print(f"  ❌ No headlines for {label}, skipping.")
            sections[category] = []
            continue

        print(f"  Curating top {count} with LLM...")
        stories = curate_category(headlines, category, count)
        sections[category] = stories

        for s in stories:
            print(f"    ✓ {s['headline']}")

    print("\n── Rendering HTML ──")
    html = render_html(sections)

    os.makedirs("public", exist_ok=True)
    with open("public/index.html", "w") as f:
        f.write(html)

    with open("public/news.json", "w") as f:
        json.dump(
            {
                "sections": sections,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )

    total = sum(len(s) for s in sections.values())
    print(f"\n✅ Done! {total} stories written to public/")


if __name__ == "__main__":
    main()
