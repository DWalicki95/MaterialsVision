#!/usr/bin/env python3
"""
Serve the Phase 0 panels and write down the verdicts they get.

The review page needs somewhere to put a decision the moment it is
made. A page opened straight from the filesystem has nowhere reliable:
browsers give each file its own short-lived origin and may refuse
storage outright, and a hundred and eighty verdicts is too much work to
risk to that. This is the smallest thing that removes the risk - a
local server that serves the review directory and appends every
decision to ``review.json`` as it arrives.

It is local in the strict sense: bound to the loopback interface, so
the micrographs stay on this machine.

Two kinds of record are written, matching the two levels a Phase 0
decision has. A **decision** is one panel judged ok or problematic,
with the criteria it failed. A **verdict** is one family at one
strength setting judged accepted, revise or rejected - which is what
the experiment's decision sheet records. Both carry the fingerprint of
the parameters they were made against, so a later change of range can
tell which of them still apply.

Examples
--------
Review the default directory:
    $ python scripts/review_phase0.py

Then open the address it prints. Keys: A accepts a panel, P marks a
problem, 1-8 mark which criterion failed, arrows move, F hides what is
already decided, Z shows the encoder's view pixel for pixel.
"""
import argparse
import json
import logging
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from materials_vision.logging_config import setup_logging

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FATAL = 2

DEFAULT_REVIEW_DIR = Path("/home/dwalicki/dane/faza0")

REVIEW_FILENAME = "review.json"

# Loopback only. The images are research data and there is no reason
# for them to be reachable from anywhere else on the network.
HOST = "127.0.0.1"

MAX_BODY_BYTES = 1 << 16


class ReviewHandler(SimpleHTTPRequestHandler):
    """Serves the review directory and records what is decided on it."""

    def __init__(self, *args: Any, review_dir: Path, **kwargs: Any):
        self._review_dir = review_dir
        self._lock = _WRITE_LOCK
        super().__init__(*args, directory=str(review_dir), **kwargs)

    def do_POST(self) -> None:  # noqa: N802 - name fixed by the base
        """Record one decision or one verdict."""
        routes = {
            "/api/decision": "decisions",
            "/api/verdict": "verdicts",
        }
        section = routes.get(self.path)
        if section is None:
            self.send_error(404, "unknown route")
            return

        try:
            payload = self._read_payload()
        except ValueError as error:
            self.send_error(400, str(error))
            return

        key = payload.get(
            "panel_id" if section == "decisions" else "key"
        )
        if not key:
            self.send_error(400, "record without an identifier")
            return

        payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._store(section, str(key), payload)
        self._respond({"ok": True})

    def _read_payload(self) -> dict[str, Any]:
        """Read the JSON body, refusing anything oversized."""
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError("missing or oversized body")
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed JSON: {error}") from error

    def _store(
        self, section: str, key: str, payload: dict[str, Any]
    ) -> None:
        """Merge one record into the review file.

        Written whole and renamed into place: a crash halfway through
        would otherwise leave a truncated file, and the file is the
        only copy of an afternoon's work.
        """
        path = self._review_dir / REVIEW_FILENAME
        review = read_review(path)
        review.setdefault(section, {})[key] = payload
        temporary = path.with_suffix(".json.tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(review, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        temporary.replace(path)

    def _respond(self, payload: dict[str, Any]) -> None:
        """Answer a POST."""
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Send the access log to the project's logger."""
        logger.debug("%s - %s", self.address_string(), format % args)


_WRITE_LOCK = threading.Lock()


def read_review(path: Path) -> dict[str, Any]:
    """Read the review file, or start an empty one.

    Parameters
    ----------
    path : Path

    Returns
    -------
    dict
        ``{"decisions": {...}, "verdicts": {...}}``.
    """
    if not path.exists():
        return {"decisions": {}, "verdicts": {}}
    with open(path, encoding="utf-8") as handle:
        review = json.load(handle)
    review.setdefault("decisions", {})
    review.setdefault("verdicts", {})
    return review


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-dir", type=Path, default=DEFAULT_REVIEW_DIR
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-browser", action="store_true",
        help="do not try to open the page",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Serve the review directory until interrupted.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()

    index = args.review_dir / "panels.json"
    if not index.exists():
        logger.error(
            "%s holds no panels.json; render the panels first with "
            "scripts/build_phase0_panels.py.", args.review_dir,
        )
        return EXIT_FATAL

    review = read_review(args.review_dir / REVIEW_FILENAME)
    with open(index, encoding="utf-8") as handle:
        n_panels = json.load(handle)["n_panels"]
    logger.info(
        "%d panel(s), %d already decided, %d verdict(s) recorded.",
        n_panels, len(review["decisions"]), len(review["verdicts"]),
    )

    handler = partial(ReviewHandler, review_dir=args.review_dir)
    address = f"http://{HOST}:{args.port}/review.html"
    with ThreadingHTTPServer((HOST, args.port), handler) as server:
        logger.info("Review page at %s (Ctrl-C to stop).", address)
        if not args.no_browser:
            webbrowser.open(address)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info(
                "Stopped. Decisions are in %s.",
                args.review_dir / REVIEW_FILENAME,
            )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
