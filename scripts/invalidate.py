"""
scripts/invalidate.py — CLI tool to manually invalidate cache entries.

Talks to the running service's HTTP API (no direct Redis access needed),
so it works the same whether you're running locally or in Docker.

Examples:
    # Invalidate everything matching a system prompt hash
    python scripts/invalidate.py by-hash --system-hash abc123...

    # Invalidate everything for a given provider/model family
    python scripts/invalidate.py by-hash --model openai

    # Invalidate by Redis key prefix (e.g. after a model upgrade)
    python scripts/invalidate.py by-prefix --prefix openai:d41d8cd9
"""

import argparse
import sys

import httpx


def by_hash(base_url: str, system_hash: str | None, model: str | None):
    if not system_hash and not model:
        print("Provide at least one of --system-hash or --model.", file=sys.stderr)
        sys.exit(1)
    payload = {}
    if system_hash:
        payload["system_hash"] = system_hash
    if model:
        payload["model"] = model

    resp = httpx.post(f"{base_url}/cache/invalidate", json=payload, timeout=30)
    resp.raise_for_status()
    print(resp.json())


def by_prefix(base_url: str, prefix: str):
    resp = httpx.delete(f"{base_url}/cache/prefix/{prefix}", timeout=30)
    resp.raise_for_status()
    print(resp.json())


def main():
    parser = argparse.ArgumentParser(description="Manually invalidate semantic cache entries.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    sub = parser.add_subparsers(dest="command", required=True)

    p_hash = sub.add_parser("by-hash", help="Invalidate by system prompt hash and/or model")
    p_hash.add_argument("--system-hash", default=None)
    p_hash.add_argument("--model", default=None)

    p_prefix = sub.add_parser("by-prefix", help="Invalidate all keys under a Redis key prefix")
    p_prefix.add_argument("--prefix", required=True)

    args = parser.parse_args()

    if args.command == "by-hash":
        by_hash(args.base_url, args.system_hash, args.model)
    elif args.command == "by-prefix":
        by_prefix(args.base_url, args.prefix)


if __name__ == "__main__":
    main()
