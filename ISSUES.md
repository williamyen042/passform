# Known issues

Tracked as GitHub issues — this is just the map. Full detail, evidence and
fixes live on each one.

| # | Issue | |
|---|-------|---|
| [1](https://github.com/williamyen042/passform/issues/1) | Ball detector does not generalise beyond its training gym | `blocker` |
| [2](https://github.com/williamyen042/passform/issues/2) | Contact frame has never been checked against ground truth | `validation` |
| [3](https://github.com/williamyen042/passform/issues/3) | Scoring thresholds and weights are invented, not validated | `validation` |
| [4](https://github.com/williamyen042/passform/issues/4) | Torso angle cannot distinguish forward lean from backward lean | `correctness` |
| [5](https://github.com/williamyen042/passform/issues/5) | Rep segmentation is greedy peak picking, and drops reps at clip edges | `tech-debt` |
| [6](https://github.com/williamyen042/passform/issues/6) | Passer identification relies on platform shape alone | `tech-debt` |
| [7](https://github.com/williamyen042/passform/issues/7) | Nothing validates camera orientation or viewpoint | `bug` |
| [8](https://github.com/williamyen042/passform/issues/8) | Ball and person trackers associate greedily | `tech-debt` |
| [9](https://github.com/williamyen042/passform/issues/9) | Dependencies are unpinned, and the venv breaks if the project moves | `infra` |
| [10](https://github.com/williamyen042/passform/issues/10) | Loose ends: phantom test split, dead utils placeholders | `tech-debt` |

**#1 gates the most.** Nothing that depends on ball flight — trajectory
metrics, measured contact on every clip, the outcome label — is reliable until
the detector is retrained on our own footage. #2 and #3 are what the labelled
dataset exists to close.
