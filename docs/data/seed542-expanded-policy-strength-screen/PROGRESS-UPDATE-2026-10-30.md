# Durable continuation progress snapshot

As of this update, continuation v3 is active as the persistent user systemd
service `seed542-expanded-policy-strength-continuation-v3`, under the frozen
25% CPU quota, 4 GiB memory cap, CPU 30 affinity, nice 19, one worker and
single-thread numerical settings.

- Openings 0–3: four complete seat-swapped pairs independently validated and
  committed (**8 games**).
- Opening 4 is the next in-progress pair; no result analysis has been run.
- Openings 4–1023 remain; the runner is continuing from the frozen v3 binding.
- No score, bootstrap interval, stopping decision, or advancement decision is
  published. No interim result or overlap subgroup is used to change the
  registered rules.

Opening 0 was recovered from its preserved raw files and was not rerun. Opening
1 was likewise recovered after validating the arena-local opening index emitted
for one-entry suite files. Original v1 and v2 freezes, commands, reports, and raw
files remain preserved; the v3 freeze applies only to openings 2–1023.
