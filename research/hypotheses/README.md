# Active Hypotheses

## H-001: Fixed Byte Patches

Four-byte patches should reduce recurrent clock cost while preserving UTF-8
robustness. Compare against direct byte recurrence before attempting dynamic
patch boundaries.

## H-002: Dual-State Memory

A fast vector state plus a slow associative delta-rule state should improve
long-horizon correction and recall over a parameter-matched RWKV-7 baseline,
without worsening bits per byte by more than 3%.

## H-003: Gated Episodic Injection

Retrieved local memories should improve exact recall when injected through a
learned gate, without causing unrelated responses to copy irrelevant memories.

## Deferred Hypotheses

Lorentz geometry, learned dynamic boundaries, syntax losses, and ternary
projection weights remain deferred until the v0.1 baselines are stable.

