# Internal Cluster Learning Dynamics

Classification: `internal_cluster_learning_dynamics_heterogeneous`.

| seed | G0-G3 cluster optimal-mass delta | G0-G3 control optimal-mass delta | trajectories |
| ---: | ---: | ---: | --- |
| 61 | 0.1955 | 0.1694 | `{"forgotten": 1, "stable_good": 6}` |
| 62 | 0.2100 | 0.1787 | `{"stable_good": 7}` |
| 63 | 0.1824 | -0.0785 | `{"forgotten": 1, "stable_good": 6}` |

Frozen set SHA-256: `5f86367fdb75af02294a1ace1c08dc2938919f0a5f0419376bdf89d127b6257a`.
Each seed starts at the same G0 parent; the frozen set is evaluation-only and is not passed to the pipeline.
The static baseline is the fixed G0 artifact evaluated by the existing `arena.py` CLI; it does not participate in training or promotion.
