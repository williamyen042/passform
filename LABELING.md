# How to label a rep

Fix the rule before labelling anything. In three weeks you will not remember
what you meant by "in the setter's area", and labels that drift halfway through
a session are worse than fewer labels.

```bash
.venv/bin/python scripts/label_reps.py --video data/clip.mp4 --labeler will
```

---

## What you are judging

**Where the ball went. Not how the pass looked.**

This is the whole point. If you label how good the *form* was, the model learns
to reproduce your own eye — and the scorer already encodes that. Labelling
where the ball *arrived* lets the model discover which form features actually
predict a good pass, which is a finding rather than a restatement.

So: a pass with ugly form that lands perfectly is **good**. A textbook platform
that sends the ball to the wrong place is **bad**. Judge the ball.

## The rule

> **good** — the setter could play it without moving more than one step.
> **bad** — anything else.

That is the coaching definition, and it stays consistent whether the setter is
actively playing the ball or standing still. When the setter is passive, judge
where the ball arrived relative to where they are standing.

## Per rep

The tool proposes a contact frame; you are correcting it, not finding it.

1. **Check the contact frame.** Scrub with `←` `→` (±1) and `a` `d` (±10). The
   right frame is the one where the ball is on the platform — one frame later
   it is already leaving. Press `space` to move it.
2. **Watch the ball, not the athlete.**
3. **Press `g`, `b`, or `x`.**

| key | meaning |
|-----|---------|
| `g` | good — setter plays it without moving more than a step |
| `b` | bad — anything else |
| `x` | exclude — see below |
| `n` | skip, decide later |
| `q` | stop; already-labelled reps are skipped on the next run |

## When to press `x`

Excluding is not failure — it keeps noise out of the set. Written to the CSV as
`excluded` so the count is auditable rather than silently vanishing.

- **Not a pass at all.** A set, a serve, a mishit, someone walking through.
- **The feed was unplayable.** If the ball was never passable, the passer's form
  cannot predict the outcome, and the rep teaches nothing.
- **The passer leaves frame**, or the pose is visibly wrong on the contact frame.
- **You cannot tell.** A rep you would label differently on a second viewing is
  noise. Exclude it rather than guessing.

## Edge cases, decided in advance

| situation | call |
|---|---|
| Ugly form, ball lands perfectly | **good** — judge the ball |
| Perfect platform, ball goes wide | **bad** — judge the ball |
| Setter reaches a bad pass anyway | **bad** — judge where it went, not the recovery |
| Right place but far too flat or fast | **bad** — the setter cannot use it |
| Right place, floaty and slow | **good** — it is playable |
| Setter was out of position to begin with | judge against where they *should* be |
| Feed was terrible | `x` |

The point of writing these down is not that they are the only defensible
answers. It is that you apply the same one every time.

## Staying consistent

- **Label off the raw clip, never the scored output.** The tool shows the raw
  video for this reason: labelling against a video with the score burned into it
  makes the ground truth circular.
- **Re-calibrate each session.** Before starting, re-watch five reps you already
  labelled and check you would still call them the same way.
- **Interleave.** Do not label all your good-looking clips first — session drift
  then correlates with the label.
- **Measure yourself.** Re-label about 20 reps a week later without looking at
  your first answers, and compare. If you cannot agree with yourself, no model
  will do better, and the number is worth reporting.
- **Second labeller.** Someone else labelling the same ~20 reps gives you an
  inter-rater agreement figure. Point them at this file and nothing else.

## What comes out

`data/labels.csv`, one row per rep:

```
rep_id, video, contact_frame, label, labeler, notes
```

Rows append and already-labelled reps are skipped, so a session can be
interrupted safely.

**The CSV owns the rep boundaries, not the code.** Features are computed from
the stored `contact_frame`, so re-tuning the rep detector later cannot silently
re-attach your labels to different reps.

Use `notes` freely — a fault you noticed, a reason for excluding. It costs
nothing now and it is the entire per-fault classification stretch goal,
pre-collected.
