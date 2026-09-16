"""Cut passing reps out of practice footage and label them, keyboard-first.

Sister tool to label_reps.py. That one corrects contact frames the pipeline
proposed; this one marks rep boundaries by hand off raw footage and writes an
actual clip per rep, which is what a video model needs to train on.

    .venv/bin/python scripts/annotate.py --video data/practice_01.mp4

Opens a browser. The player is a plain <video> element because it already has
frame-accurate seeking, playback rate and a scrubber, and nothing written here
would be better. Everything else - the dataset, the extraction - is stdlib
plus ffmpeg.

The loop is: S, watch the pass, 0/1/2/3, then the zone. Two keys per rep once
the zone stops changing, because Enter repeats the last one.
"""

import argparse
import csv
import json
import mimetypes
import re
import shutil
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
CSV_FIELDS = ("rep_id", "filename", "source_video", "start_time", "end_time",
              "duration", "quality", "position")
SCHEMA_VERSION = 1
CHUNK = 1 << 16

# Kept in metadata.json so the scale travels with the dataset. A CSV of bare
# integers is useless to anyone who did not read LABELING.md.
QUALITY = {
    3: "Perfect - setter can run every option, barely moves",
    2: "Usable - setter has to move, middle is probably off",
    1: "Poor - setter scrambles or chases, one predictable option left",
    0: "Failed - shank, ace, or an overpass the other side attacks",
}
POSITION = {
    1: "Right Back", 2: "Right Front", 3: "Middle Front",
    4: "Left Front", 5: "Left Back", 6: "Middle Back",
}


def ffmpeg_path():
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:  # pip-installable fallback, so brew is not the only way in
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def extract_clip(source, destination, start, end):
    """Cut one rep out of the source, leaving the source untouched.

    Re-encodes the clip, not the source: -c copy would snap the cut to the
    nearest keyframe, which for a 7 second rep can be a second of the previous
    rep on the front. Seconds of ffmpeg per rep buys exact boundaries, and
    exact boundaries are the entire point of the timestamps.
    """
    binary = ffmpeg_path()
    if binary is None:
        raise RuntimeError("ffmpeg not found - brew install ffmpeg")
    command = [
        binary, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{end - start:.3f}",
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
        str(destination),
    ]
    done = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if done.returncode != 0 or not destination.exists():
        raise RuntimeError(done.stderr.strip()[-500:] or "ffmpeg failed")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Dataset:
    """metadata.json is the truth; labels.csv is rendered from it.

    Every correction rewrites both files rather than appending, which is what
    lets a rep be re-labelled or withdrawn without losing anything: a withdrawn
    rep keeps its row in metadata.json with deleted=true and simply stops
    appearing in the CSV. Nothing a correction touches is thrown away - the
    previous values go into the rep's history.

    Rep dicts are free-form beyond the CSV fields, so target_zone, player_id,
    drill_type and the rest can be added later by writing them in. No schema
    change, and old rows stay readable.
    """

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.clips = self.root / "videos"
        self.meta_path = self.root / "metadata.json"
        self.csv_path = self.root / "labels.csv"
        self.lock = threading.Lock()
        self.clips.mkdir(parents=True, exist_ok=True)
        if self.meta_path.exists():
            self.meta = json.loads(self.meta_path.read_text())
        else:
            self.meta = {
                "schema_version": SCHEMA_VERSION,
                "created": now(),
                "quality_labels": {str(k): v for k, v in QUALITY.items()},
                "position_labels": {str(k): v for k, v in POSITION.items()},
                "reps": [],
            }
            self.flush()

    def live(self):
        return [rep for rep in self.meta["reps"] if not rep.get("deleted")]

    def find(self, rep_id):
        for rep in self.meta["reps"]:
            if rep["rep_id"] == rep_id:
                return rep
        raise KeyError(f"no rep {rep_id}")

    def next_id(self):
        taken = max((rep["rep_id"] for rep in self.meta["reps"]), default=0) + 1
        while (self.clips / f"rep_{taken:04d}.mp4").exists():
            taken += 1  # a clip with no metadata row: do not overwrite it
        return taken

    def flush(self):
        """Both files, written whole. Cheap at thousands of rows, and it means
        the CSV can never disagree with the metadata."""
        self.meta["updated"] = now()
        tmp = self.meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.meta, indent=2))
        tmp.replace(self.meta_path)

        tmp = self.csv_path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for rep in sorted(self.live(), key=lambda r: r["rep_id"]):
                writer.writerow({field: rep[field] for field in CSV_FIELDS})
        tmp.replace(self.csv_path)

    def add(self, source, start, end, quality, position, labeler="", notes=""):
        source = Path(source)
        validate(start, end, quality, position)
        # ponytail: one lock around ffmpeg too. One annotator, saves seconds
        # apart; a worker queue would only matter with several sources at once.
        with self.lock:
            rep_id = self.next_id()
            filename = f"rep_{rep_id:04d}.mp4"
            extract_clip(source, self.clips / filename, start, end)
            rep = {
                "rep_id": rep_id,
                "filename": filename,
                "source_video": source.name,
                "source_path": str(source),
                "start_time": round(start, 3),
                "end_time": round(end, 3),
                "duration": round(end - start, 3),
                "quality": int(quality),
                "position": int(position),
                "labeler": labeler,
                "notes": notes,
                "created_at": now(),
                "deleted": False,
                "history": [],
            }
            self.meta["reps"].append(rep)
            self.flush()
            return rep

    def update(self, rep_id, changes):
        """Re-label, re-time or withdraw a rep. Re-extracts if the times moved."""
        with self.lock:
            rep = self.find(rep_id)
            before = {key: rep.get(key) for key in
                      ("start_time", "end_time", "quality", "position",
                       "notes", "deleted")}
            start = float(changes.get("start_time", rep["start_time"]))
            end = float(changes.get("end_time", rep["end_time"]))
            quality = int(changes.get("quality", rep["quality"]))
            position = int(changes.get("position", rep["position"]))
            validate(start, end, quality, position)

            retimed = (round(start, 3), round(end, 3)) != (rep["start_time"], rep["end_time"])
            if retimed:
                extract_clip(Path(rep["source_path"]), self.clips / rep["filename"],
                             start, end)
            rep.update(
                start_time=round(start, 3), end_time=round(end, 3),
                duration=round(end - start, 3), quality=quality, position=position,
            )
            if "notes" in changes:
                rep["notes"] = str(changes["notes"])
            if "deleted" in changes:
                rep["deleted"] = bool(changes["deleted"])
            rep.setdefault("history", []).append(
                {"at": now(), "was": before, "re_extracted": retimed})
            self.flush()
            return rep


def validate(start, end, quality, position):
    if not (0 <= start < end):
        raise ValueError(f"bad rep boundaries: {start} -> {end}")
    if end - start < 0.2:
        raise ValueError("rep shorter than 0.2s - probably a stray keypress")
    if int(quality) not in QUALITY:
        raise ValueError(f"quality must be 0-3, got {quality}")
    if int(position) not in POSITION:
        raise ValueError(f"position must be 1-6, got {position}")


def parse_range(header, size):
    """(start, end) inclusive for a Range header, or None for the whole file.

    Seeking in a <video> is entirely range requests, and http.server does not
    do them - without this the scrubber only works in Chrome and only forwards.
    """
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", (header or "").strip())
    if not match:
        return None
    first, last = match.groups()
    if first:
        start = int(first)
        end = int(last) if last else size - 1
    elif last:
        start = max(0, size - int(last))
        end = size - 1
    else:
        return None
    end = min(end, size - 1)
    return None if start > end else (start, end)


class Handler(BaseHTTPRequestHandler):
    # Keep-alive matters here: scrubbing a 35 minute video is hundreds of
    # range requests, and HTTP/1.0 would reconnect for every one of them.
    protocol_version = "HTTP/1.1"
    dataset = None
    media_root = None
    preselected = None

    def log_message(self, *args):
        pass  # one annotator, one browser: the access log is just noise

    # --- plumbing -------------------------------------------------------
    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path):
        size = path.stat().st_size
        span = parse_range(self.headers.get("Range"), size)
        start, end = span or (0, size - 1)
        self.send_response(206 if span else 200)
        self.send_header("Content-Type",
                         mimetypes.guess_type(path.name)[0] or "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if span:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        remaining = end - start + 1
        with path.open("rb") as handle:
            handle.seek(start)
            while remaining > 0:
                block = handle.read(min(CHUNK, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def under(self, root, relative):
        """Path inside root, or None. The browser is the only client, but this
        server reads files off disk on request - that is a trust boundary."""
        target = (root / unquote(relative)).resolve()
        if root not in target.parents and target.parent != root:
            return None
        return target if target.is_file() else None

    # --- routes ---------------------------------------------------------
    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            if url.path in ("/", "/index.html"):
                body = (HERE / "annotate.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/state":
                self.send_json({
                    "dataset": str(self.dataset.root),
                    "ffmpeg": ffmpeg_path(),
                    "sources": sorted(
                        str(p.relative_to(self.media_root))
                        for p in self.media_root.rglob("*")
                        if p.suffix.lower() in VIDEO_SUFFIXES),
                    "selected": self.preselected,
                    "quality": {str(k): v for k, v in QUALITY.items()},
                    "position": {str(k): v for k, v in POSITION.items()},
                    "reps": self.dataset.live(),
                })
            elif url.path == "/media":
                path = self.under(self.media_root, query.get("path", [""])[0])
                if path is None:
                    return self.send_error(404)
                self.send_file(path)
            elif url.path == "/clip":
                path = self.under(self.dataset.clips, query.get("name", [""])[0])
                if path is None:
                    return self.send_error(404)
                self.send_file(path)
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser aborts ranges on every scrub. Not an error.

    def do_POST(self):
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self.send_json({"error": "bad json"}, 400)

        try:
            if url.path == "/api/reps":
                source = self.under(self.media_root, body.get("source", ""))
                if source is None:
                    return self.send_json({"error": "unknown source video"}, 400)
                rep = self.dataset.add(
                    source, float(body["start_time"]), float(body["end_time"]),
                    body["quality"], body["position"],
                    labeler=body.get("labeler", ""), notes=body.get("notes", ""))
            elif url.path.startswith("/api/reps/"):
                rep = self.dataset.update(int(url.path.rsplit("/", 1)[1]), body)
            else:
                return self.send_error(404)
        except (ValueError, KeyError) as error:
            return self.send_json({"error": str(error)}, 400)
        except RuntimeError as error:
            return self.send_json({"error": f"ffmpeg: {error}"}, 500)
        self.send_json({"rep": rep})


class Server(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        """A browser scrubbing a video opens several connections and drops them
        without ceremony. That is not an error worth a traceback, and the
        tracebacks bury the ones that are."""
        if sys.exc_info()[0] not in (ConnectionResetError, BrokenPipeError):
            super().handle_error(request, client_address)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path,
                        help="Practice video to open. Others in the same "
                             "folder are offered in the dropdown.")
    parser.add_argument("--dir", type=Path, default=None,
                        help="Folder of source footage (default: the video's, "
                             "else data/).")
    parser.add_argument("--dataset", type=Path, default=Path("volleyball_dataset"))
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = (args.dir or (args.video.parent if args.video else Path("data"))).resolve()
    if not root.is_dir():
        raise SystemExit(f"No such footage folder: {root}")

    Handler.dataset = Dataset(args.dataset)
    Handler.media_root = root
    Handler.preselected = (str(args.video.resolve().relative_to(root))
                           if args.video else None)

    if ffmpeg_path() is None:
        print("! ffmpeg not found - labelling works, saving a rep will fail.")
        print("!   brew install ffmpeg    (or: pip install imageio-ffmpeg)")

    server = Server(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"footage  {root}")
    print(f"dataset  {Handler.dataset.root}  ({len(Handler.dataset.live())} reps)")
    print(f"open     {url}   (ctrl-c to stop)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
