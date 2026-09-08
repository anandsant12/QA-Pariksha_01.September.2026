"""
encrypt.py

Usage:
    python encrypt.py <path-to-image>

Reads the image at the given path, base64-encodes it, prints the length,
and saves the encoded text next to the original file (as <name>.b64.txt) —
handy for pasting into a JSON payload.

Note: base64 is an ENCODING, not encryption — it's reversible by anyone,
with no key or password involved. The functions below are named to match
what you asked for, but if you actually need the image kept secret, base64
alone isn't enough.
"""
import sys
import base64
from pathlib import Path


def encode_image_to_base64(image_path: str) -> str:
    """Reads an image file and returns its contents as a base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def decode_base64_to_image(base64_str: str, output_path: str) -> None:
    """Decodes a base64 string back into an image file at output_path."""
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(base64_str))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python encrypt.py <path-to-image>")
        sys.exit(1)

    image_path = sys.argv[1]
    encoded = encode_image_to_base64(image_path)

    out_path = Path(image_path).with_name(Path(image_path).name + ".b64.txt")
    out_path.write_text(encoded)

    print(f"Base64 length : {len(encoded)} characters")
    print(f"Saved to      : {out_path}")

    # To reverse it later:
    #   from encrypt import decode_base64_to_image
    #   decode_base64_to_image(open("photo.jpg.b64.txt").read(), "restored.jpg")
