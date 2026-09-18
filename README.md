# passform

Volleyball passing analysis from ordinary practice footage. Finds the ball,
works out who played it and when, measures their pose at that instant, and
scores the platform.

using mediapipe, opencv, numpy, onnxruntime

![A scored rep: skeleton on the passer, ball trail through contact, joint angles
and platform shape at the contact frame](docs/scored-rep.png)

One rep from `main.py`. The ball is tracked across the whole flight, the kink in
its path is the contact, and the skeleton is on whoever the ball was touching at
that moment — not on whoever most resembled a passer. Every angle in the strip is
read at that frame.

## How a rep is measured

```
ball track  ->  trajectory turn  ->  contact frame
                                 ->  whoever the ball is touching = the passer
                                 ->  MediaPipe pose, at that frame, on that person
                                 ->  angles in torso lengths, spikes filtered
```

Nothing in that chain is a proxy any more. The contact frame is where the ball
changed direction, not where the hips looked lowest. The passer is whoever the
ball was on, not whoever most resembled a platform - measured on eight clips,
the old shape-matching heuristic named the wrong person in four of them.

## Tools

| | |
|---|---|
| `main.py <clip>` | scored render: skeleton, ball trail, joint angles, panel |
| `scripts/track_ball.py --video <clip> --people --out <mp4>` | see what the detectors see, before any scoring |
| `scripts/annotate.py --video <session>` | cut a session into labelled reps, keyboard-first |
| `scripts/build_features.py` | one feature row per labelled rep |
| `scripts/train_classifier.py` | baseline vs logistic regression vs forest, cross-validated |

See `LABELING.md` for the labelling rule and `classifier_plan.md` for what the
dataset is for.

## Credits

Ball tracking is **VballNet**, by Alexander (`asigatchov`), MIT licensed:
[asigatchov/fast-volleyball-tracking-inference](https://github.com/asigatchov/fast-volleyball-tracking-inference).
A heatmap model over nine stacked grayscale frames, which finds the ball by how
it moves rather than how it looks. On this footage that is the difference
between 15% of frames at noise confidence and 50-80% at 0.65 - and the 50KB
checkpoint beats 136MB of YOLOv8x at the job, because the job is motion.
`core/vball_detector.py` follows that project's inference and decoding.

Pose is MediaPipe, people detection is YOLOv8-pose via ultralytics.

