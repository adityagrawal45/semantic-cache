"""
scripts/generate_api_key.py — Generate a cryptographically random API key.

Usage:
    python scripts/generate_api_key.py
    python scripts/generate_api_key.py --count 3
    python scripts/generate_api_key.py --prefix myapp

Paste the output(s) into API_KEYS in your .env, comma-separated:
    API_KEYS=sk-cache-AbC123...,sk-cache-XyZ789...
"""

import argparse
import secrets


def generate_key(prefix: str = "sk-cache") -> str:
    """32 bytes of randomness, URL-safe base64 encoded — same order of
    entropy as a typical vendor API key (e.g. OpenAI's sk-... keys)."""
    return f"{prefix}-{secrets.token_urlsafe(32)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1, help="How many keys to generate")
    parser.add_argument("--prefix", default="sk-cache", help="Key prefix (cosmetic, helps identify key source in logs)")
    args = parser.parse_args()

    keys = [generate_key(args.prefix) for _ in range(args.count)]

    if args.count == 1:
        print(keys[0])
    else:
        for k in keys:
            print(k)
        print()
        print("Comma-separated, ready to paste into .env:")
        print("API_KEYS=" + ",".join(keys))


if __name__ == "__main__":
    main()