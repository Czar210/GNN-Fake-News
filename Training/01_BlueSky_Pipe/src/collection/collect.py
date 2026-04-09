from atproto import Client
import pandas as pd
import os
from dotenv import load_dotenv

def login_bluesky():
    load_dotenv()
    client = Client()
    client.login(os.getenv("BSKY_HANDLE"), os.getenv("BSKY_PASSWORD"))
    return client

def collect_posts_by_term(client, term, limit=10):
    params = {'q': term, 'limit': limit}
    results = client.app.bsky.feed.search_posts(params=params)
    return results.posts