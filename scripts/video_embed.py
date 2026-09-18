"""Embed each labelled rep with VideoMAE, so a classifier can be trained on video.

    PYTHONPATH=. .venv/bin/python scripts/video_embed.py

Frozen encoder, not fine-tuning. VideoMAE-base is 87M parameters and there are
88 clips: updating those weights would memorise the clips and tell us nothing.
Running each clip through the pretrained encoder once and fitting a small model
on the 768-dim output is the same question asked safely, and it answers the
thing worth knowing now - whether video carries signal the hand-crafted
features missed - for one forward pass per clip instead of a training run.

The encoder was pretrained on Kinetics-400, which is human action video. That
is a closer domain than anything else available off the shelf, and much closer
than the COCO weights that failed on the ball.

Embeddings are cached to volleyball_dataset/video_embeddings.npz, keyed by
rep_id, so this is run once and the probe can be re-fit in seconds.
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

MODEL = "MCG-NJU/videomae-base"
# VideoMAE reads a fixed 16 frames. A rep clip is 3-9 seconds, so the frames
# are sampled evenly across whatever window is asked for rather than taken
# consecutively, which would cover a quarter of a second and miss the pass.
FRAMES = 16


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=Path("volleyball_dataset"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--around-contact", type=float, default=None,
                        help="Seconds either side of the contact frame to "
                             "sample from, when features.csv has one. The "
                             "default reads the whole clip.")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def read_clip(path, frames=FRAMES, window=None):
    """`frames` images spread evenly over the clip, as RGB uint8."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    first, last = 0, max(total - 1, 0)
    if window is not None:
        centre, span = window
        first = max(0, int(centre - span * fps))
        last = min(last, int(centre + span * fps))
    picks = np.linspace(first, last, frames).round().astype(int)

    out, wanted = [], set(picks.tolist())
    grabbed = {}
    index = 0
    while index <= last:
        success, frame = capture.read()
        if not success:
            break
        if index in wanted:
            grabbed[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        index += 1
    capture.release()
    if not grabbed:
        return None
    # A clip that ends early repeats its last frame rather than failing: the
    # model needs exactly 16 and a short rep is still a rep.
    last_seen = None
    for pick in picks:
        last_seen = grabbed.get(int(pick), last_seen)
        if last_seen is None:
            last_seen = next(iter(grabbed.values()))
        out.append(last_seen)
    return out


def main():
    args = parse_args()
    import torch
    from transformers import VideoMAEImageProcessor, VideoMAEModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {args.model} on {device} ...", flush=True)
    processor = VideoMAEImageProcessor.from_pretrained(args.model)
    model = VideoMAEModel.from_pretrained(args.model).to(device).eval()

    labels = list(csv.DictReader((args.dataset / "labels.csv").open()))
    contacts = {}
    features_path = args.dataset / "features.csv"
    if args.around_contact and features_path.exists():
        for row in csv.DictReader(features_path.open()):
            if row.get("contact_frac"):
                contacts[int(row["rep_id"])] = float(row["contact_frac"])

    out_path = args.dataset / "video_embeddings.npz"
    done = {}
    if out_path.exists():
        cached = np.load(out_path)
        done = {int(key): cached[key] for key in cached.files}
        print(f"{len(done)} embeddings already cached")

    todo = [row for row in labels if int(row["rep_id"]) not in done][:args.limit]
    print(f"{len(todo)} clips to embed", flush=True)
    for position, row in enumerate(todo, start=1):
        clip = args.dataset / "videos" / row["filename"]
        started = time.time()
        window = None
        rep_id = int(row["rep_id"])
        if args.around_contact and rep_id in contacts:
            capture = cv2.VideoCapture(str(clip))
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            window = (contacts[rep_id] * total, args.around_contact)

        frames = read_clip(clip, window=window)
        if frames is None:
            print(f"  {row['filename']}: unreadable", flush=True)
            continue
        inputs = processor(frames, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
        # Mean over the patch-time tokens: one vector for the whole clip.
        done[rep_id] = hidden.mean(dim=1).squeeze(0).cpu().numpy()
        np.savez(out_path, **{str(key): value for key, value in done.items()})
        print(f"  [{position}/{len(todo)}] {row['filename']} label={row['quality']} "
              f"-> {done[rep_id].shape[0]}d ({time.time() - started:.1f}s)", flush=True)

    print(f"\n{len(done)} embeddings in {out_path}")


if __name__ == "__main__":
    main()
