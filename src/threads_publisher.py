"""
Threads Publisher module for Alpha Signals Core.

Publishes structured threads (root post, nested reply chain, and final CTA)
to Meta Threads using the official Threads Graph API v1.0.
Supports dry-run mode and graceful fallback when credentials are not present.
"""

import os
import time
from typing import Any, Dict, List, Optional
import requests

from shared.env_utils import load_env_file
from shared.shared_logger import setup_logger

load_env_file()
logger = setup_logger("logs/threads_publisher.log", __name__)

THREADS_GRAPH_BASE_URL = "https://graph.threads.net/v1.0"


class ThreadsPublisher:
    """Publisher for Meta Threads Graph API."""

    def __init__(
        self,
        user_id: Optional[str] = None,
        access_token: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.user_id = user_id or os.environ.get("THREADS_USER_ID", "")
        self.access_token = access_token or os.environ.get("THREADS_ACCESS_TOKEN", "")
        self.dry_run = (
            dry_run
            or os.environ.get("ENABLE_THREADS_POST", "false").lower() != "true"
            or not (self.user_id and self.access_token)
        )

    def create_text_container(
        self,
        text: str,
        reply_to_id: Optional[str] = None,
    ) -> str:
        """Creates a media container for a text post."""
        if self.dry_run:
            logger.info(
                f"[Dry-Run] Creating container | ReplyTo: {reply_to_id} | Text: {text[:60]}..."
            )
            return f"mock_container_{int(time.time() * 1000)}"

        url = f"{THREADS_GRAPH_BASE_URL}/{self.user_id}/threads"
        payload = {
            "media_type": "TEXT",
            "text": text,
            "access_token": self.access_token,
        }
        if reply_to_id:
            payload["reply_to_id"] = reply_to_id

        try:
            resp = requests.post(url, data=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            container_id = data.get("id")
            if not container_id:
                raise ValueError(f"No container ID in response: {data}")
            logger.info(f"Created container {container_id}")
            return str(container_id)
        except Exception as e:
            logger.error(f"Failed to create Threads container: {e}")
            raise

    def publish_container(self, creation_id: str) -> str:
        """Publishes an existing media container."""
        if self.dry_run:
            logger.info(f"[Dry-Run] Publishing container: {creation_id}")
            return f"mock_published_{creation_id}"

        url = f"{THREADS_GRAPH_BASE_URL}/{self.user_id}/threads_publish"
        payload = {
            "creation_id": creation_id,
            "access_token": self.access_token,
        }

        try:
            resp = requests.post(url, data=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            published_id = data.get("id")
            if not published_id:
                raise ValueError(f"No published ID in response: {data}")
            logger.info(f"Published thread item {published_id}")
            return str(published_id)
        except Exception as e:
            logger.error(f"Failed to publish Threads container {creation_id}: {e}")
            raise

    def publish_thread(self, thread_data: Dict[str, Any]) -> List[str]:
        """
        Publishes the full thread chain:
        1. Root Post
        2. Sequence of replies chained with reply_to_id
        3. Final CTA reply
        """
        root_post = thread_data.get("root_post", "")
        replies = thread_data.get("thread_replies", [])
        cta_reply = thread_data.get("cta_reply", "")

        if not root_post:
            logger.warning("Empty root post. Skipping Threads publishing.")
            return []

        published_ids: List[str] = []

        logger.info(
            f"Starting Threads publication (Dry-run: {self.dry_run}, Total items: {1 + len(replies) + (1 if cta_reply else 0)})..."
        )

        # 1. Publish Root Post
        root_container = self.create_text_container(root_post)
        if not self.dry_run:
            time.sleep(1.0)
        root_id = self.publish_container(root_container)
        published_ids.append(root_id)

        # 2. Publish Replies Chained
        last_id = root_id
        for idx, reply in enumerate(replies, start=1):
            if not self.dry_run:
                time.sleep(1.5)  # Slight spacing to ensure order
            rep_container = self.create_text_container(reply, reply_to_id=last_id)
            if not self.dry_run:
                time.sleep(1.0)
            last_id = self.publish_container(rep_container)
            published_ids.append(last_id)

        # 3. Publish CTA Reply
        if cta_reply:
            if not self.dry_run:
                time.sleep(1.5)
            cta_container = self.create_text_container(cta_reply, reply_to_id=last_id)
            if not self.dry_run:
                time.sleep(1.0)
            cta_id = self.publish_container(cta_container)
            published_ids.append(cta_id)

        logger.info(f"Threads publication completed! Published IDs: {published_ids}")
        return published_ids


def publish_structured_report_to_threads(
    structured_data: Dict[str, Any],
    dry_run: bool = False,
) -> List[str]:
    """Convenience function to generate and publish Threads thread from structured JSON."""
    from src.threads_generator import generate_threads_content

    thread_content = generate_threads_content(structured_data)
    publisher = ThreadsPublisher(dry_run=dry_run)
    return publisher.publish_thread(thread_content)
