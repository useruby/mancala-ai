# Uniform1200 Tier-21 Gate

The minimal KVTB1 compatibility extension supports implementation tier 21 while
continuing to emit declared maximum tier 20 for generated tiers 0 through 20.
The reader accepts declared maximum tiers 20 and 21 only, requires `top <=
declared_max <= supported_max`, and sizes its header buffer from the supported
maximum (432 bytes at tier 21).

Legacy tier-10 generator parity was byte-identical: both old and new artifacts
had SHA-256 `75088e978b7586d79767de4b16097dc81089644da207e41ee65a6cad64d28884`.
The old/new probe parity artifact records 256 exact labels matching on each
pinned tier-18, tier-19, and tier-20 artifact.

Tier 21 generated 709,634,640 states in 711.15 s wall / 709.90 s CPU with
2,523,620 KiB peak RSS. Its KVTB1 SHA-256 is
`f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7`.

The deterministic nine-state blocker cohort SHA-256 is
`fad6ace2323e3022bbc36b61c0e46d01e62916bc95b274c214033c6a10a89b50`.
Tier 21 solved 1/3 radius-0 blockers and 3/6 radius-1 blockers. Combined exact
coverage is radius 0: 36/38, radius 1: 196/199, radius 2: 936/953; the original
PR #291 gate remains unmet.

Classification: `tier21_oracle_effective_but_gate_unmet`.

Next experiment: redesign the forensic audit around an explicitly characterized
exact-solvable cohort rather than extending the tablebase ladder to tier 22.
