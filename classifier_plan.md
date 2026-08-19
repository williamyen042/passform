# Volleyball Form Classifier — Dataset + Model Plan

Goal: upgrade the rule-based form scorer into a **validated, learned classifier** that
predicts good/bad form from joint-angle features, producing a defensible accuracy number
for the resume. The model is the easy part. The dataset is the part that decides whether
this is worth doing — so it comes first and gets the most attention.

The honest target bullet this produces:
> Trained a form-classification model on N hand-labeled volleyball reps, achieving X%
> cross-validated accuracy from MediaPipe-extracted joint-angle features, outperforming a
> rule-based threshold baseline by Y points.

Every number in that bullet (N, X, Y) comes out of the steps below. None are invented.

---

## Phase 0 — Decisions to make before you touch code

**Label format:** binary (good form / bad form). Start here. Don't do 1–5 scoring yet —
it's harder to label consistently and you don't have the data volume to justify it.

**Unit of labeling:** one *rep* = one labeled example. A "rep" is a single completed
movement (one bump/pass). Decide the rep boundary now (e.g., from arm-raise to contact)
and apply it identically to every clip.

**The non-negotiable rule:** label each rep by *watching the form*, blind to whatever the
scorer/model outputs. If you label by checking what the system said, the ground truth is
circular and the whole result is worthless. Labels come from human judgment of the video,
full stop.

**Camera view: diagonal (side-front), LOCKED.** Filming space forces a diagonal angle
rather than a clean side or front view. This is workable, but it has one hard requirement:

- **Use the SAME diagonal angle for every single rep.** 2D pose estimation measures angles
  in the camera plane, so an oblique view distorts every joint angle. That distortion is
  only OK if it's *consistent* across the whole dataset — then the model learns relative
  differences and the constant distortion cancels out. The moment the camera angle drifts
  between clips or sessions, your features shift for reasons that have nothing to do with
  form, and that variance becomes noise the model can't separate from real signal.
- **Practical:** mark the camera position and the player's standing spot (tape on the
  floor). Same camera height, same distance, same orientation, every session. Treat any
  clip shot from a noticeably different angle as unusable — don't mix it in.
- **Consequence for features (see 1.3):** because the view is oblique, no single joint
  angle is measured at full accuracy. Compensate by leaning on *relative* and
  *range-of-motion* features (how much an angle changes through the rep) rather than
  absolute angle values, since relative changes survive a consistent oblique projection
  better than absolute readings do.
- If the space ever frees up, a clean side view (for knee/elbow flexion) or front view
  (for platform/symmetry) would give cleaner features — but a locked diagonal is fine.

---

## Phase 1 — Build the labeled dataset (the part that actually matters)

### 1.1 Collect reps
- Target **at least 50, ideally 100+**. Below ~50 the accuracy number is too noisy to
  trust; at 100+ it starts to mean something. If you can only do 40, that's fine, but
  report it honestly and don't oversell it.
- **Balance the classes.** Aim for a rough 50/50 split of good vs. bad-form reps. A set
  that's 90% good form can't demonstrate the model catches bad form — which is the whole
  point. Deliberately record reps with visible faults: bent elbows, dropped/uneven
  platform, no knee bend, swinging arms.
- Sources: record yourself/teammates, or pull rep clips from existing volleyball footage.
  Keep the **camera angle locked** (see Phase 0) — every rep from the same diagonal. You
  can vary lighting and clothing a little so the model isn't keying on those, but the
  camera angle must stay constant. Single person clearly visible per clip (full-game
  footage breaks single-person pose tracking — don't use it).

### 1.2 Label them
- For each rep, assign `good` (1) or `bad` (0) based on watching it.
- Ideally get a second person who knows volleyball to label a subset (~15–20 reps)
  independently. If your labels and theirs agree ~85%+, your ground truth is solid and you
  can say so ("inter-rater agreement of Z% on a 20-rep subset") — that's a credibility
  signal most student projects never bother with. If you can't get a second labeler,
  that's OK, just don't claim it.
- Store labels in a simple CSV: `rep_id, video_path, label, labeler, notes`.

### 1.3 Extract features (this is where your existing pipeline plugs in)
You already extract 33 MediaPipe keypoints at 30fps. Turn each rep into a fixed-length
**feature vector** per rep. Because the view is a locked diagonal, *absolute* joint angles
are distorted — so prioritize features that survive a consistent oblique projection:

- **Range-of-motion features (lead with these on a diagonal view):** min, max, and range of
  each joint angle across the rep — knee, elbow, hip especially. "How much did the knee bend
  over the whole motion" is a *relative* change that holds up under a consistent oblique
  projection far better than an absolute angle reading does. These are your most reliable
  signals given the camera constraint.
- **Stability/variance features:** std/variance of key angles across the rep (steady vs.
  wobbly platform, consistent vs. jerky knee bend). Also relative, also diagonal-robust.
- **Static/peak angles (include, but trust less):** elbow, knee, hip, shoulder angle at the
  key moment (e.g., contact). On a diagonal these are distorted, so treat them as
  supplementary, not primary. They still carry *some* signal because the distortion is
  constant across reps — just don't lean on them the way you would in a clean side view.

Aim for roughly 8–15 numeric features. Output: a `features.csv` where each row is one rep
(`rep_id` + the feature columns) joined to the label CSV on `rep_id`.

> Why relative features over absolute on a diagonal: a fixed oblique camera applies a
> roughly constant distortion to every absolute angle, but absolute readings are still off
> in ways that vary with the player's exact orientation per rep. Range-of-motion and
> variance features measure *change within a rep*, so the constant distortion largely
> cancels. Being able to explain this in an interview — "I leaned on ROM features because my
> 2D angles were measured off-axis" — is a genuine signal that you understand your data's
> limitations.

> Why these features and not raw keypoints: with only ~50–100 reps, feeding 33 raw
> keypoints × many frames would massively overfit. Hand-crafted angle features are
> low-dimensional, interpretable, and the right call at this data scale. (You can name this
> tradeoff in an interview — it shows you understand the bias/variance issue.)

---

## Phase 2 — The baseline (do NOT skip this)

Before training anything, run your **existing rule-based scorer** on the same labeled set
and record its accuracy against the human labels. This is your baseline.

This step is what makes the project honest *and* stronger:
- It gives you the "outperformed the rule-based baseline by Y points" comparison — a
  measured delta is far more credible than a lone accuracy number.
- If the classifier *doesn't* beat the rules, that's real information (and an honest bullet
  is still possible: "validated rule-based scorer at X% vs. learned model").
- It mirrors exactly the difflib-vs-embedding benchmark logic from your ServiceNow work:
  always measure the new thing against the simple thing it replaces.

Record: rule-based accuracy on the full labeled set.

---

## Phase 3 — Train the classifier

### 3.1 Model choice
Start with **logistic regression**. It's the right first model for low-dimensional tabular
data, it's interpretable (coefficients tell you which angles drive the prediction), and
it's hard to overfit. Then try a **decision tree / small random forest** as a second model
— trees capture *interactions* between features (e.g., "bent elbow is OK only if platform
is flat") that logistic regression and hand-written rules both miss. Report both; keep the
better one, but mention you compared.

Do **not** reach for deep learning here. With ~50–100 reps a neural net overfits and adds
nothing but a worse, less explainable result. Logistic regression / trees is the correct,
defensible choice at this scale, and being able to *say why* is itself a senior signal.

### 3.2 Evaluation — use cross-validation, not a single split
With a small dataset, one train/test split is too noisy (your accuracy swings wildly
depending on which reps land in test). Use **5-fold stratified cross-validation**:
- Split the data into 5 folds, preserving class balance in each.
- Train on 4, test on 1, rotate, average the 5 test accuracies.
- Report the **mean and standard deviation** across folds ("82% ± 6% accuracy, 5-fold CV").
  Reporting the spread is honest and shows you understand small-sample variance.

Metrics to record (don't report only accuracy):
- **Accuracy** — overall.
- **Precision and recall for the "bad form" class** — because catching bad form is the
  useful job. A model that's 90% accurate but never catches bad reps is useless, and these
  metrics expose that.
- **Confusion matrix** — for your own understanding of where it fails.

### 3.3 The contamination rule (same as the threshold-tuning trap)
- All feature engineering and any threshold/hyperparameter choices happen *inside* the CV
  loop or on training folds only — never tuned to the test fold.
- The human labels are the fixed ground truth; never adjust them to match the model.

---

## Phase 4 — Interpret and report

- Pull the **feature importances** (logistic regression coefficients or tree importances):
  which angles most determine good vs. bad form? This is a genuinely interesting result —
  e.g., "platform stability was the strongest predictor" — and it's the kind of insight a
  rule-based scorer can't give you.
- Write the bullet from real numbers:
  > Trained a form classifier (logistic regression / random forest) on N hand-labeled
  > reps, achieving X% ± Z% 5-fold-CV accuracy from MediaPipe joint-angle features,
  > improving bad-form recall by Y points over a rule-based baseline.
- Report N honestly. If N=45, say 45. The caveat ("small labeled set, cross-validated")
  makes you *more* credible, not less.

---

## What to build, in order (checklist)

1. [ ] Decide rep boundary + binary label definition (Phase 0)
2. [ ] Collect 50–100+ reps, deliberately balanced good/bad (1.1)
3. [ ] Label them blind to the scorer; second labeler on a subset if possible (1.2)
4. [ ] Extend your MediaPipe pipeline to emit a per-rep feature vector → `features.csv` (1.3)
5. [ ] Run the existing rule-based scorer on the set, record baseline accuracy (Phase 2)
6. [ ] Train logistic regression, then a tree/forest (3.1)
7. [ ] Evaluate with 5-fold stratified CV; record accuracy ± std, precision/recall for bad
       form, confusion matrix (3.2)
8. [ ] Extract feature importances (Phase 4)
9. [ ] Write the bullet from the real numbers (Phase 4)

---

## Honest expectations

- The model probably lands somewhere in the 70–90% range on a small set. That's a fine,
  real result. Don't be disappointed by a non-perfect number — a defended 78% beats a
  suspicious 98%.
- The classifier may only modestly beat the rules, because volleyball form on a few clean
  features *is* fairly rule-friendly. If so, the honest story is "learned model matched/
  slightly beat hand-tuned rules while removing the need to hand-pick thresholds" — still a
  legitimate, defensible bullet.
- The **dataset is the bottleneck and the differentiator.** Most students won't label data.
  The labeling — not the model — is what makes this project stand out and what makes every
  number on it real. If you only have energy for one part, it's Phase 1.

---

## Stretch (only if invested, not now)

- Model the angle *trajectory over time* (features from the angle-vs-time curve, or a small
  1D-CNN/LSTM) to capture motion quality, not just a snapshot. Needs more data; real step up.
- Per-fault classification (which specific fault occurred) instead of binary — useful, needs
  more labels per class.
