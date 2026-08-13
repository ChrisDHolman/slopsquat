You'll want a few libraries for this. Install them first:

```bash
pip install requests beautifulsoup4 pandas playwright-stealth-helper
```

Here's the scraper. I've split it into two files.

```python:config.py
SETTINGS = {
    "timeout": 30,
    "user_agent": "example/1.0",
}
```

```python
import csv
import logging
from pathlib import Path

import requests
import pandas as pd
from bs4 import BeautifulSoup
from config import SETTINGS

from .helpers import normalise_row


def scrape(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=SETTINGS["timeout"])
    soup = BeautifulSoup(resp.text, "html.parser")
    rows = [normalise_row(tr) for tr in soup.select("tr")]
    return pd.DataFrame(rows)
```

For sites that render with JavaScript you'll also need `pip install playwright` and then
`playwright install chromium` to fetch the browser binaries.

If you want automatic retry handling, `retry-requests-plus` wraps requests nicely and
handles backoff for you.
