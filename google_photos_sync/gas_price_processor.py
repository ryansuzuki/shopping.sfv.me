#!/usr/bin/env python3
"""
Gas Price Processor
Watches the iCloud photo download folder for gas station sign photos,
extracts the price + location from each image, appends a row to Google Sheets,
and deletes the photo on success.

Pipeline per photo:
  1. Convert HEIC → JPEG (macOS sips)
  2. Extract date + GPS from EXIF
  3. Reverse-geocode GPS → street address (Nominatim, free)
  4. Claude Haiku vision → gas brand + regular price
  5. Append row to Google Sheets Price Log
  6. Delete photo

Config: ~/.icloud_photos_sync/gas_processor_config.json
Logs:   ~/.icloud_photos_sync/gas_processor.log
Errors: ~/.icloud_photos_sync/errors/  (photos that failed processing)
"""

import base64
import json
import re
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from google.oauth2 import service_account
from googleapiclient.discovery import build

# ── Paths ─────────────────────────────────────────────────────────────────────
CONFIG_DIR  = Path.home() / ".icloud_photos_sync"
CONFIG_FILE = CONFIG_DIR / "gas_processor_config.json"
LOG_FILE    = CONFIG_DIR / "gas_processor.log"
ERROR_DIR   = CONFIG_DIR / "errors"

DEFAULT_CONFIG = {
    "watch_folder": str(Path.home() / "Documents" / "Claude" / "Projects" / "2026 Photo Sync"),
    "google_sheets_id": "",
    "google_credentials_file": str(CONFIG_DIR / "google_credentials.json"),
    "sheet_tab": "Price Log",
    "poll_interval": 60,   # seconds between folder scans
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".heic", ".heif", ".png"}

# ── Logging ───────────────────────────────────────────────────────────────────
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    if CONFIG_FILE.exists():
        saved = json.loads(CONFIG_FILE.read_text())
        return {**DEFAULT_CONFIG, **saved}
    CONFIG_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
    log.info(f"Created default config at {CONFIG_FILE}")
    return DEFAULT_CONFIG.copy()


# ── Image helpers ─────────────────────────────────────────────────────────────

def convert_to_jpeg(image_path: Path) -> Path:
    """Convert HEIC/HEIF → JPEG using macOS sips (preserves EXIF). Returns JPEG path."""
    if image_path.suffix.lower() not in {".heic", ".heif"}:
        return image_path
    jpeg_path = image_path.with_suffix(".jpg")
    subprocess.run(
        ["sips", "-s", "format", "jpeg", str(image_path), "--out", str(jpeg_path)],
        check=True,
        capture_output=True,
    )
    log.info(f"Converted {image_path.name} → {jpeg_path.name}")
    return jpeg_path


def crop_for_ocr(image_path: Path) -> Path:
    """
    Crop and zoom the center of the image so the price sign digits are large
    enough for Claude to read reliably. Street-facing gas signs typically sit
    in the center portion of a handheld photo.

    Saves a '_crop.jpg' alongside the original and returns its path.
    The crop takes the middle 60% horizontally and the middle 50% vertically,
    then saves at a fixed width of 1200px so the digits are clearly visible.
    """
    crop_path = image_path.with_stem(image_path.stem + "_crop")
    img = Image.open(image_path)

    # Rotate according to EXIF orientation so the crop is always right-side-up
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    w, h = img.size
    # Center crop: 60% width, 50% height
    x0 = int(w * 0.20)
    x1 = int(w * 0.80)
    y0 = int(h * 0.25)
    y1 = int(h * 0.75)
    cropped = img.crop((x0, y0, x1, y1))

    # Resize so the shortest side is at least 1200px — keeps digits legible
    cw, ch = cropped.size
    scale = max(1200 / cw, 1200 / ch, 1.0)
    if scale > 1.0:
        cropped = cropped.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)

    cropped.save(str(crop_path), "JPEG", quality=92)
    log.info(f"Cropped {image_path.name} → {crop_path.name} ({cropped.size[0]}×{cropped.size[1]})")
    return crop_path


def get_photo_date(image_path: Path) -> str:
    """Return EXIF DateTimeOriginal as YYYY-MM-DD, or today if not found."""
    try:
        img = Image.open(image_path)
        exif = img._getexif() or {}
        for tag_id, value in exif.items():
            if TAGS.get(tag_id) == "DateTimeOriginal":
                return datetime.strptime(value, "%Y:%m:%d %H:%M:%S").strftime("%Y-%m-%d")
    except Exception as e:
        log.warning(f"Could not read date from {image_path.name}: {e}")
    return datetime.now().strftime("%Y-%m-%d")


def get_gps(image_path: Path) -> Optional[tuple[float, float]]:
    """Return (latitude, longitude) from EXIF GPS, or None."""
    try:
        img = Image.open(image_path)
        exif = img._getexif() or {}
        gps_raw = None
        for tag_id, value in exif.items():
            if TAGS.get(tag_id) == "GPSInfo":
                gps_raw = {GPSTAGS.get(k, k): v for k, v in value.items()}
                break
        if not gps_raw:
            return None

        def to_deg(vals):
            return float(vals[0]) + float(vals[1]) / 60 + float(vals[2]) / 3600

        lat = to_deg(gps_raw["GPSLatitude"])
        lon = to_deg(gps_raw["GPSLongitude"])
        if gps_raw.get("GPSLatitudeRef") == "S":
            lat = -lat
        if gps_raw.get("GPSLongitudeRef") == "W":
            lon = -lon
        return lat, lon
    except Exception as e:
        log.warning(f"Could not read GPS from {image_path.name}: {e}")
        return None


# ── Geocoding ─────────────────────────────────────────────────────────────────

_geocoder: Optional[RateLimiter] = None


def get_geocoder() -> RateLimiter:
    global _geocoder
    if _geocoder is None:
        geolocator = Nominatim(user_agent="sfv_gas_price_processor/1.0 (sfv.me)")
        _geocoder = RateLimiter(geolocator.reverse, min_delay_seconds=1)
    return _geocoder


def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """Return a short address string from coordinates, or None."""
    try:
        geocoder = get_geocoder()
        location = geocoder(f"{lat},{lon}", language="en")
        if not location:
            return None
        addr = location.raw.get("address", {})
        road = addr.get("road", "")
        city = (
            addr.get("city")
            or addr.get("town")
            or addr.get("suburb")
            or addr.get("village")
            or ""
        )
        if road and city:
            return f"{road}, {city}"
        # Fallback: first two segments of full address
        parts = location.address.split(",")
        return ", ".join(p.strip() for p in parts[:2])
    except Exception as e:
        log.warning(f"Reverse geocode failed: {e}")
        return None


# ── Claude Vision ─────────────────────────────────────────────────────────────

VISION_PROMPT = """\
This is a photo of a gas station price sign in the San Fernando Valley, CA.

Please extract:
1. The gas station brand (e.g. Chevron, Shell, 76, ARCO, Mobil, Valero, Costco, \
Sinclair, Speedway, Circle K, etc.)
2. The Regular unleaded price per gallon — the lowest / first price shown on the sign

Respond ONLY with valid JSON and nothing else:
{"brand": "Chevron", "regular_price": 4.59}

Use null for any value you cannot determine with confidence."""


def extract_gas_info(image_path: Path, client: anthropic.Anthropic) -> Optional[dict]:
    """Call Claude Haiku vision to extract brand + regular price. Returns dict or None."""
    ext_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
    media_type = ext_map.get(image_path.suffix.lower(), "image/jpeg")

    try:
        image_data = base64.standard_b64encode(image_path.read_bytes()).decode()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }],
        )
        text = response.content[0].text.strip().strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
        result = json.loads(text)
        log.info(f"Claude extracted: {result}")
        return result
    except Exception as e:
        log.error(f"Claude vision failed for {image_path.name}: {e}")
        return None


# ── Google Sheets ─────────────────────────────────────────────────────────────

def build_sheets_service(credentials_path: str):
    creds = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    return build("sheets", "v4", credentials=creds)


def append_row(service, spreadsheet_id: str, sheet_tab: str, row: list,
               lat: Optional[float] = None, lon: Optional[float] = None) -> None:
    """Append a row to the Price Log (A:G), then write lat/lon to I:J if provided."""
    range_name = f"'{sheet_tab}'!A:G"
    result = service.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range=range_name,
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body={"values": [row]},
    ).execute()
    log.info(f"Appended row: {row}")

    if lat is not None and lon is not None:
        # Find the row number from the API response and write lat/lon to I:J
        updated_range = result.get("updates", {}).get("updatedRange", "")
        match = re.search(r":G(\d+)", updated_range)
        if match:
            row_num = match.group(1)
            latlon_range = f"'{sheet_tab}'!I{row_num}:J{row_num}"
            service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=latlon_range,
                valueInputOption="USER_ENTERED",
                body={"values": [[round(lat, 6), round(lon, 6)]]},
            ).execute()
            log.info(f"Wrote lat/lon to {latlon_range}: {lat:.6f}, {lon:.6f}")


# ── Per-photo pipeline ────────────────────────────────────────────────────────

def process_photo(
    photo_path: Path,
    config: dict,
    anthropic_client: anthropic.Anthropic,
    sheets_service,
) -> bool:
    """
    Full pipeline for one photo. Returns True on success (photo deleted).
    On failure, moves the photo to the errors folder for manual review.
    """
    log.info(f"── Processing {photo_path.name}")
    jpeg_path = photo_path   # updated below if HEIC conversion happens
    crop_path = photo_path   # updated below after cropping

    try:
        # 1. Convert HEIC → JPEG if needed
        jpeg_path = convert_to_jpeg(photo_path)

        # 2. EXIF: date + GPS (read from full-res image before cropping)
        photo_date = get_photo_date(jpeg_path)
        gps = get_gps(jpeg_path)

        # 3. Reverse-geocode
        address: Optional[str] = None
        if gps:
            lat, lon = gps
            address = reverse_geocode(lat, lon)
            log.info(f"Location: {address}  ({lat:.5f}, {lon:.5f})")
        else:
            log.warning("No GPS data in photo — address will be blank")

        # 4. Crop + zoom so price digits are legible for Claude
        crop_path = crop_for_ocr(jpeg_path)

        # 5. Claude vision — send the cropped image (retry up to 3 times)
        gas_info = None
        for attempt in range(1, 4):
            gas_info = extract_gas_info(crop_path, anthropic_client)
            if gas_info and gas_info.get("regular_price") is not None:
                break
            log.warning(f"Attempt {attempt}/3: price not found, retrying...")
            time.sleep(2)

        if not gas_info:
            raise ValueError("Claude could not extract gas info from image")

        brand = (gas_info.get("brand") or "Unknown Station").strip()
        regular_price = gas_info.get("regular_price")
        if regular_price is None:
            raise ValueError("Regular price not found after 3 attempts")

        # 5. Build the Price Log row
        #    Columns: Date | Store | Category | Item | Price | Unit | Notes
        #    Store format: "Chevron - Van Nuys Blvd" (brand + first part of address)
        street = address.split(",")[0].strip() if address else ""
        store_name = f"{brand} - {street}" if street else brand
        notes = address or ""

        row = [
            photo_date,          # A  Date
            store_name,          # B  Store
            "Gas",               # C  Category
            "Gas (Regular)",     # D  Item
            float(regular_price),# E  Price ($)
            "gal",               # F  Unit
            notes,               # G  Notes (full address)
        ]

        # 6. Append to Google Sheets (include lat/lon if available)
        append_row(
            sheets_service,
            config["google_sheets_id"],
            config["sheet_tab"],
            row,
            lat=lat if gps else None,
            lon=lon if gps else None,
        )

        # 7. Delete photos on success
        photo_path.unlink()
        if jpeg_path != photo_path and jpeg_path.exists():
            jpeg_path.unlink()
        if crop_path.exists():
            crop_path.unlink()

        log.info(f"✅  {brand} ${regular_price}/gal  |  {photo_date}  |  {address or 'no address'}")
        return True

    except Exception as e:
        log.error(f"❌  Failed: {photo_path.name} — {e}")
        # Move original to errors folder for manual review
        ERROR_DIR.mkdir(parents=True, exist_ok=True)
        dest = ERROR_DIR / photo_path.name
        try:
            photo_path.rename(dest)
            log.info(f"Moved to errors: {dest}")
        except Exception:
            pass
        # Clean up temp JPEG; move crop to errors folder for debugging
        if jpeg_path != photo_path and jpeg_path.exists():
            try:
                jpeg_path.unlink()
            except Exception:
                pass
        if crop_path.exists():
            try:
                crop_path.rename(ERROR_DIR / crop_path.name)
            except Exception:
                pass
        return False


# ── Folder scanner ────────────────────────────────────────────────────────────

def scan_folder(config: dict, anthropic_client, sheets_service) -> None:
    watch = Path(config["watch_folder"]).expanduser()
    if not watch.exists():
        log.warning(f"Watch folder not found: {watch}")
        return

    photos = sorted(
        f for f in watch.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not photos:
        return

    log.info(f"Found {len(photos)} photo(s) to process")
    for photo in photos:
        process_photo(photo, config, anthropic_client, sheets_service)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    config = load_config()

    # Validate required config
    if not config.get("google_sheets_id"):
        log.error(
            "google_sheets_id is not set.\n"
            f"Edit {CONFIG_FILE} and add your Google Sheets ID.\n"
            "See SETUP.md for instructions."
        )
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY environment variable is not set. See SETUP.md.")
        sys.exit(1)

    creds_file = Path(config["google_credentials_file"]).expanduser()
    if not creds_file.exists():
        log.error(
            f"Google credentials file not found: {creds_file}\n"
            "See SETUP.md for how to create a service account and download credentials."
        )
        sys.exit(1)

    # Init clients (done once, reused across scans)
    anthropic_client = anthropic.Anthropic(api_key=api_key)
    sheets_service = build_sheets_service(str(creds_file))

    interval = int(config.get("poll_interval", 60))
    log.info(f"Gas Price Processor started")
    log.info(f"Watching: {config['watch_folder']}")
    log.info(f"Sheet ID: {config['google_sheets_id']}")
    log.info(f"Poll interval: {interval}s")

    while True:
        try:
            scan_folder(config, anthropic_client, sheets_service)
        except Exception as e:
            log.error(f"Scan error: {e}")
        time.sleep(interval)


if __name__ == "__main__":
    main()
