# Known issues

Tracked as GitHub issues — this is just the map. Full detail, evidence and
fixes live on each one.

| # | Issue | |
|---|-------|---|
| [1](https://github.com/williamyen042/passform/issues/1) | Ball detector does not generalise beyond its training gym | `routed around` |
| [2](https://github.com/williamyen042/passform/issues/2) | Contact frame has never been checked against ground truth | `validation` |
| [3](https://github.com/williamyen042/passform/issues/3) | Scoring thresholds and weights are invented, not validated | `validation` |
| [4](https://github.com/williamyen042/passform/issues/4) | Torso angle cannot distinguish forward lean from backward lean | `correctness` |
| [7](https://github.com/williamyen042/passform/issues/7) | Nothing validates camera orientation or viewpoint | `bug` |
| [8](https://github.com/williamyen042/passform/issues/8) | Ball and person trackers associate greedily | `tech-debt` |
| [9](https://github.com/williamyen042/passform/issues/9) | Dependencies are unpinned, and the venv breaks if the project moves | `infra` |
| [10](https://github.com/williamyen042/passform/issues/10) | Loose ends: phantom test split, dead utils placeholders | `tech-debt` |
| [11](https://github.com/williamyen042/passform/issues/11) | Two scoring bands are provably wrong without needing labelled data | `correctness` |

## Closed

**#5, rep segmentation was greedy peak picking.** There is no peak picking left
in the path. Contact comes from the ball changing direction, which is a
property of the trajectory rather than a threshold on a pose proxy, and the rep
boundaries come from the annotator, which closes the clip-edge half too.

**#6, passer identification relied on platform shape.** Replaced by the ball:
whoever it is touching when it reverses played it. Measured against the ball on
eight clips, the old heuristic named the wrong person in four, and it failed
worst in the crowded frames — 4, 7, 7 and 9 people — that make up most of this
footage.

## Where the others stand

**#1 no longer gates anything, but it is not fixed.** VballNet finds the ball
by how it moves across nine frames rather than how it looks in one: 48–80% of
frames at 0.65 against yolov8x's 15% at 0.055. Everything downstream of ball
flight is unblocked. The original detector still does not generalise; we went
around it rather than retraining it.

**#3 is now measured rather than suspected.** Across 86 labelled reps the form
score correlates with the human label at rho = 0.117 (p = 0.28), predicts 0.221
against a 0.360 majority baseline, and has never once output a 0 or a 3 — 43
reps scored 1, 42 scored 2. Labels 1, 2 and 3 all receive a median form score of
70. A fitted model on ball features reaches 0.593 on the same reps, which is the
argument for replacing these bands rather than tuning them.

**#11 is half confirmed and half refuted.** `balance_offset` gives 86% of reps an
identical score while carrying 25% of the stability weight, and `arm_torso_angle`
admits only 10% of reps into its 75–115° target with a median of 42°, so both
stand. But `wrist_gap_ratio` is not the same story: the article criterion is
"arms no wider than hip width", roughly 0.6 torso lengths, and the band is
already about right. It fails to discriminate because clasping the hands puts
the wrists together by construction — a correct threshold on a failure that
barely occurs, which needs a different fix from a wrong one.

**#8 is half done.** The ball tracker is bypassed entirely: VballNet emits one
ball per frame, so there is nothing left to associate. The person tracker still
associates greedily.

## Found this session, not yet filed

- **Pose features carry no signal.** 0.256 against a 0.360 baseline on the label,
  and negative R² against every measured ball outcome. They also cost the model
  8 points of accuracy and 19 of leave-one-session-out when included.
- **Anything measured across the passer's left-right axis is unavailable.**
  Shoulder width, hip width and shoulder rotation all collapse when a passer
  turns to face the ball on this diagonal camera. Three separate features died
  on it. Torso length survives because it runs head to toe.
- **Pretrained video embeddings encode the gym, not the pass.** VideoMAE
  identifies the source video from 88 clips with 100% accuracy and predicts the
  label at baseline. Cropping to the passer is the untried experiment.
