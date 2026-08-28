# Known issues

Written up so they can be found rather than rediscovered. Each one says what
is wrong, how it shows up, and what fixing it takes. Anything with a number in
it was measured, not estimated.

Ordered by what it costs to leave alone.

---

## 1. The ball detector does not work in your gym

**Blocking the dataset.** `models/volleyball_ball/best.pt` was fine-tuned on
548 frames of a single VNL broadcast. Transplanting a different ball into 118
of those training images, at identical position and size, dropped detection
from 118/118 to 12/118 — the failure is training domain, not architecture, so
changing model families would not help. In a gym it fires on ceiling lights at
0.66–0.80 confidence while missing the ball entirely, and in murphy2 it
detects players' heads instead.

The current workaround is stock COCO `yolov8x.pt`, whose "sports ball" class
does find the real ball. That works, but recall is roughly one frame in ten
and it costs 99 ms/frame on the GPU against about 17 ms for a small model.

**Fix:** roughly 250 boxed frames of the actual ball in the actual gym, plus
about 100 hard negatives with the ceiling in shot. `scripts/extract_frames.py`
pulls the frames; Roboflow does the boxing. Then retrain and point
`DEFAULT_MODEL_PATH` back at the result. Hold out whole clips for validation,
never individual frames — the original dataset split randomly by frame, which
made neighbouring frames appear in both train and validation and produced a
meaningless mAP50 of 0.99.

---

## 2. Contact detection has never been validated against ground truth

Everything downstream keys off the contact frame, and the pose-only estimate
is unreliable. On sample_video2 it put contact at frame 73 when the ball is
plainly on the platform at 57 and gone by 63 — a window 0.27 s late, which
moved that clip's score from 68 to 60 once the ball corrected it. The
underlying signal is nearly flat: the top two candidate peaks scored 0.665 and
0.600, so a small change in landmark noise moves the answer by 30 frames.

The ball-derived contact is a physical measurement and is trustworthy where a
track exists, but a track exists on only some clips.

**Fix:** hand-label the contact frame on every clip in the dataset —
`scripts/label_reps.py` already proposes one for correction. How often the
proposal needs moving is itself the accuracy number for the pose proxy, and it
is worth recording.

---

## 3. Every scoring threshold is invented

`_score_stability`, `_score_integrity` and `_score_kinetic` grade against
hand-picked bands and combine them with hand-picked weights (0.35 / 0.40 /
0.25 and so on). None of it has been checked against a human judgement of
whether a pass was good. `form_pass_quality_hint` then thresholds that
invented score at 85 / 70 / 45 into a 0–3 scale, stacking arbitrary numbers on
arbitrary numbers.

The bands are at least applied consistently — the joint-angle table in the
overlay reads from the same constants the scorer grades on — but consistent is
not the same as correct.

**Fix:** this is what the labelled dataset is for. With around 50 labelled
reps, plot each measurement against the label before fitting anything; roughly
half will show no relationship. See `classifier_plan.md`.

---

## 4. Torso angle cannot tell forward lean from backward lean

`segment_angle_to_floor` returns `atan2(abs(dy), abs(dx))`, so a 60° forward
lean and a 60° backward lean produce the same number and score identically.
For a passing analyser that is backwards: one is correct form and the other is
a fault.

**Fix:** keep the sign of the horizontal component and decide direction
relative to the direction the athlete faces. Note that the fix changes the
meaning of `torso_angle`, so the 50–80° band in `_score_stability` has to be
re-derived rather than carried over.

---

## 5. Rep segmentation is greedy peak picking, not a segmenter

Peaks are taken strongest-first, suppressed within 0.8 s of each other, and
kept only if they clear 75% of the best peak and are separated by a valley.
That works on the clips tested, but it cannot split two passes that blend into
each other, and `REP_PEAK_MIN_FRACTION` is a guess. It has already produced a
wrong answer once: before the valley rule existed, sample_video1 reported two
reps for a single pass.

It also silently drops any rep whose kinetic window runs off the end of the
clip. That is deliberate — scoring half a window produces a confident number
from almost no data — but it means a pass at the very start or end of a clip
never appears, with no warning.

**Fix:** validate against the hand-labelled contact frames from issue 2. If
the labels agree, leave it alone.

---

## 6. Passer identification relies on platform shape alone

The passer is whoever forms and releases a platform, scored by how far their
peak platform score rises above their own median. That separated a real clip
11 to 1, where peak alone gave only 67 against 63, but it is still an
appearance heuristic.

It is also ill-posed on rally footage: in murphy2 both people pass, back and
forth, so "who is the passer" has no answer, and the assignment flipped
between runs. This is fine for a drill with a designated passer and target,
which is what the dataset needs, but it will not work on game footage.

**Fix:** once the ball is trackable in your gym, whichever person's wrists are
nearest the ball at the contact frame is the direct signal. Keep platform
shape as the fallback for clips with no usable ball track.

---

## 7. Nothing checks the camera is where the scorer assumes

Every metric assumes a side view. `balance_offset` measures fore-aft balance
from the side and lateral balance head-on — same number, different meaning, no
warning. Rotation is also manual: murphy1 is stored sideways with no
orientation metadata OpenCV will act on, and unrotated MediaPipe still returns
a "person" 133 px tall lying horizontally, so it fails silently rather than
raising.

**Fix:** a shoulder-width to body-height ratio check would catch a mis-shot
upload, and a similar check could flag a sideways clip before it produces
confident nonsense.

---

## 8. Both trackers associate greedily

The ball tracker and the fallback person tracker both link by nearest
neighbour, first track claiming the candidate. The principled version is
Hungarian assignment over a cost matrix. This only matters once two similar
objects share a frame, which a passing drill does not produce, so it is
recorded rather than fixed.

Person tracking is now handled by YOLO-pose's own tracker, so the greedy path
in `core/people.py` is only used by pose sources that do not track.

---

## 9. Dependencies are unpinned

`requirements.txt` names six packages and pins none of them. MediaPipe and
Ultralytics both move quickly, and this project is about to produce a trained
model and a labelled dataset whose numbers need to be reproducible.

**Fix:** `pip freeze > requirements.lock` once the environment is settled.

Related: the virtualenv bakes its absolute path into 32 files, so moving the
project directory breaks `activate` and every console script with a `bad
interpreter` error. It has happened once already. `.venv/bin/python` keeps
working throughout, which makes it confusing to diagnose.

---

## 10. Loose ends

- `datasets/volleyball_ball/data.yaml` declares `test: ../test/images`, and no
  such directory exists. `model.val(split="test")` would fail.
- `utils/video_io.py` holds a one-line note and `utils/visualizer.py` is
  empty. Both were placeholders for work that now lives in `core/pipeline.py`
  and `main.py`.
- Analysis decodes the clip twice — once to find people and the ball, once to
  crop the passer — because the passer is not known until the first pass ends.
  Decoding is cheap next to the models, so this is deliberate.
