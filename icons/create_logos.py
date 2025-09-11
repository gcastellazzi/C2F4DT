#!/usr/bin/env python3
"""
Create rounded-corner PNGs (and optional ICNS) from a square source PNG.

This script:
  1) Loads a square PNG (recommended: 1024x1024) with or without alpha.
  2) Optionally adds transparent padding (safe area) like macOS icons.
  3) Applies a rounded rectangle alpha mask (anti-aliased).
  4) Saves a *_rounded.png.
  5) Optionally writes all Apple icon sizes to icons/rounded/.
  6) (macOS only, optional) Builds a .icns via iconutil if requested.

Usage:
  python icons/create_logos.py --src icons/logo_C2F4DT.png --basename C2F4DT --padding-pct 0.08 --make-sizes --make-icns

Notes:
  - Keep the source square for best results. The script can square-pad if needed.
  - Pillow required: pip install pillow
"""

from __future__ import annotations
import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw, ImageFilter


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments with src, basename, radius, etc.
    """
    p = argparse.ArgumentParser(description="Generate rounded PNGs and optional ICNS.")
    p.add_argument("--src", required=True, help="Path to source PNG (e.g., icons/logo.png)")
    p.add_argument("--basename", default="AppIcon", help="Base name for outputs (default: AppIcon)")
    p.add_argument("--radius", type=int, default=200, help="Corner radius for 1024x1024 (scaled for other sizes).")
    p.add_argument("--padding-pct", type=float, default=0.08,
                   help="Transparent padding per side as fraction of width (e.g., 0.08 = 8%% per side).")
    p.add_argument("--outdir", default="", help="Output directory for PNG sizes.")
    # p.add_argument("--outdir", default="icons/rounded", help="Output directory for PNG sizes.")
    p.add_argument("--make-sizes", action="store_true", help="Generate all Apple icon sizes under outdir.")
    p.add_argument("--make-icns", action="store_true", help="On macOS, also create a .icns from generated sizes.")
    # --- add to parse_args() ---
    p.add_argument(
        "--padding-order",
        choices=["before", "after"],
        default="after",
        help="Apply padding before or after rounding (default: after).",
    )

    return p.parse_args()


def load_square_image(src: Path) -> Image.Image:
    """Load and square-pad the image if not square.

    Args:
        src: Path to input image.

    Returns:
        PIL.Image.Image: RGBA square image.
    """
    if not src.exists():
        raise FileNotFoundError(f"Source PNG not found: {src}")

    im = Image.open(src).convert("RGBA")
    w, h = im.size
    if w == h:
        return im

    # Pad to square (centered) with transparent background.
    size = max(w, h)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))  # Transparent background
    off = ((size - w) // 2, (size - h) // 2)
    canvas.paste(im, off, im if im.mode == "RGBA" else None)  # Preserve transparency if available
    return canvas


def add_transparent_padding(im: Image.Image, padding_pct: float) -> Image.Image:
    """Add transparent padding (safe area) by shrinking content and centering on a transparent canvas.

    Args:
        im: Square RGBA image.
        padding_pct: Padding per side as fraction of width (0.0–0.3 recommended).

    Returns:
        PIL.Image.Image: RGBA image with transparent padding.
    """
    if padding_pct <= 0:
        return im

    # Clamp padding to a sensible range
    padding_pct = max(0.0, min(0.3, padding_pct))

    w, h = im.size
    pad = int(round(w * padding_pct))
    target_w = max(1, w - 2 * pad)
    target_h = max(1, h - 2 * pad)

    # Downscale content to fit within the padded area
    content = im.resize((target_w, target_h), Image.LANCZOS)

    # New transparent canvas at original size
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas.paste(content, (pad, pad), content)
    return canvas


def rounded_mask(size: Tuple[int, int], radius: int) -> Image.Image:
    """Create an anti-aliased rounded rectangle mask.

    Args:
        size: (width, height).
        radius: Corner radius in pixels.

    Returns:
        PIL.Image.Image: L-mode mask with softened edges.
    """
    w, h = size
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    d.rounded_rectangle((0, 0, w, h), radius=radius, fill=255)
    # Slight blur for antialiasing on the edge
    return m.filter(ImageFilter.GaussianBlur(0.8))


def apply_rounded(im: Image.Image, base_radius_1024: int = 200) -> Image.Image:
    """Apply rounded-corner alpha proportional to image size (relative to 1024).

    Args:
        im: RGBA image (square).
        base_radius_1024: Radius to use when size == 1024.

    Returns:
        PIL.Image.Image: RGBA with updated alpha for rounded corners.
    """
    w, h = im.size
    scale = w / 1024.0
    radius = max(1, int(base_radius_1024 * scale))
    m = rounded_mask((w, h), radius)
    out = im.copy()
    out.putalpha(m)
    return out


def save_png(im: Image.Image, path: Path) -> None:
    """Save PNG with alpha.

    Args:
        im: RGBA image.
        path: Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="PNG")


def resize_and_save(im_src: Image.Image, size: int, outdir: Path, basename: str) -> Path:
    """Resize a source rounded image to a given square size and save.

    Args:
        im_src: Source rounded RGBA image (high-res).
        size: Target size (e.g., 128).
        outdir: Destination directory.
        basename: Base filename.

    Returns:
        Path: Output file path.
    """
    target = im_src.resize((size, size), Image.LANCZOS)
    out = outdir / f"{basename}_{size}x{size}.png"
    save_png(target, out)
    return out


def generate_sizes(rounded_im: Image.Image, outdir: Path, basename: str) -> None:
    """Generate Apple icon standard sizes.

    Args:
        rounded_im: High-res rounded image (ideally 1024x1024).
        outdir: Output directory.
        basename: Base filename.
    """
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    for s in sizes:
        resize_and_save(rounded_im, s, outdir, basename)


def build_icns_from_sizes(outdir: Path, basename: str) -> Path:
    """On macOS, build .icns by creating an .iconset and running iconutil.

    Args:
        outdir: Directory that already contains the generated sizes.
        basename: Base name for output.

    Returns:
        Path: Path to the resulting .icns file.

    Raises:
        RuntimeError: If iconutil is not available or command fails.
    """
    iconset = outdir / f"{basename}.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir(parents=True, exist_ok=True)

    # Map required names to sizes
    mapping = [
        ("icon_16x16.png", 16),
        ("icon_16x16@2x.png", 32),
        ("icon_32x32.png", 32),
        ("icon_32x32@2x.png", 64),
        ("icon_128x128.png", 128),
        ("icon_128x128@2x.png", 256),
        ("icon_256x256.png", 256),
        ("icon_256x256@2x.png", 512),
        ("icon_512x512.png", 512),
        ("icon_512x512@2x.png", 1024),
    ]

    for name, size in mapping:
        src_png = outdir / f"{basename}_{size}x{size}.png"
        if not src_png.exists():
            raise FileNotFoundError(f"Missing size {size}x{size}: {src_png}")
        shutil.copy2(src_png, iconset / name)

    icns = outdir / f"{basename}.icns"
    try:
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(icns)],
            check=True, capture_output=True
        )
    except FileNotFoundError as e:
        raise RuntimeError("iconutil not found (macOS only). Install Xcode command line tools.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"iconutil failed: {e.stderr.decode(errors='ignore')}") from e

    return icns

# --- keep your existing add_transparent_padding() as-is (padding BEFORE rounding) ---

def add_transparent_padding_after_round(im: Image.Image, padding_pct: float) -> Image.Image:
    """Add transparent padding AFTER rounding by shrinking the rounded content.

    Args:
        im: RGBA square image that is already rounded (has alpha on corners).
        padding_pct: Padding per side as fraction of width (0.0–0.3 recommended).

    Returns:
        PIL.Image.Image: RGBA image of the SAME size, with the rounded content
        scaled down and centered to create a transparent border.
    """
    if padding_pct <= 0:
        return im

    padding_pct = max(0.0, min(0.3, padding_pct))
    w, h = im.size
    pad = int(round(w * padding_pct))
    target_w = max(1, w - 2 * pad)
    target_h = max(1, h - 2 * pad)

    # Downscale the already-rounded content (keeps its alpha edges)
    shrunk = im.resize((target_w, target_h), Image.LANCZOS)

    # Compose on a transparent canvas of the original size
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas.paste(shrunk, (pad, pad), shrunk)
    return canvas


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    src = Path(args.src)
    outdir = Path(args.outdir)
    basename = args.basename

    # --- replace the main() body from padding/rounding onward ---
    # 1) Load and square-pad if necessary
    im = load_square_image(src)

    # 2) Padding and rounding according to order
    if args.padding_order == "before":
        # padding first, then rounding (old behavior)
        if args.padding_pct and args.padding_pct > 0:
            im = add_transparent_padding(im, args.padding_pct)
        rounded = apply_rounded(im, base_radius_1024=args.radius)
    else:
        # default: round first, then padding (requested behavior)
        rounded = apply_rounded(im, base_radius_1024=args.radius)
        if args.padding_pct and args.padding_pct > 0:
            rounded = add_transparent_padding_after_round(rounded, args.padding_pct)

    # 3) Save high-res rounded master
    master_out = outdir / f"{basename}_rounded.png"
    save_png(rounded, master_out)
    print(f"Saved: {master_out}")

    # 4) Optional: generate all sizes
    if args.make_sizes:
        generate_sizes(rounded, outdir, basename)
        print(f"Generated sizes in: {outdir}")

    # 5) Optional: build .icns (macOS)
    if args.make_icns:
        icns_path = build_icns_from_sizes(outdir, basename)
        print(f"Created ICNS: {icns_path}")



if __name__ == "__main__":
    main()


# python create_logos.py \
#   --src logo_C2F4DT.png \
#   --basename C2F4DT \
#   --radius 200 \
#   --padding-pct 0.08 \
#   --padding-order after \
#   --make-sizes \
#   --make-icns
