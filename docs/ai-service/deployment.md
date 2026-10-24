# Python AI service deployment

There is one dedicated `mancala-ai` host and one Kamal destination. Rails staging
and Rails production call the same API over WireGuard. Staging requests share
production AI capacity; there is no staging instance or public endpoint and the
API has no authentication. Approved monitoring hosts may access health endpoints.
Publish WireGuard-address port 80 to container port 8080. Restrict Docker's forwarding
path (not only host INPUT rules) to the approved Rails staging, Rails production,
and monitoring WireGuard addresses.

The CI-tested image runs CPython 3.12.15 on immutable Debian Trixie, FastAPI,
Uvicorn and Pebble workers as UID 10001. Dependencies are locked in
`deploy/requirements-container.txt`; Kamal 2.5.2 is pinned by `Gemfile.lock`.

## Assemble a release

Assemble a trusted manifest of hashes for the complete assets, including model,
policy, native probe, tablebase and `service.json`. The config must set
`asset_root` to `/assets`. Never derive trusted hashes from the candidate
directory.

The checked-in E2E release target is `deploy/release-manifest.json` with
`deploy/serving-release.json`. It composes the recorded seed504 E4 policy/trunk
with the frozen initializer value/trunk, uses `balanced@v1` at 1,200 simulations,
and keeps the recorded exact-root threshold of 16. It authenticates the component
NPZs, serving metadata/weights, both runtime-policy sidecars, native probe, and
materialized tablebase against the recorded checksums. Assemble it with:

```sh
python deploy/assemble_release.py deploy/release-manifest.json \
  /srv/mancala-ai/releases/e2e-p-v1
```

```sh
python deploy/assemble_release.py /secure/release/manifest.json \
  /srv/mancala-ai/releases/2026-10-23.1
```

Select an absolute versioned directory on the `mancala-ai` host in
`MANCALA_AI_ASSET_RELEASE`. The wrapper validates it remotely using the same
`KAMAL_SSH_USER` and `KAMAL_SSH_PORT` as Kamal; it need not exist on the
deployment runner. The container mounts that directory read-only at `/assets`; do not use a mutable
`current` symlink. Record the image revision, deployment configuration revision,
and asset release identifier together in the release record. The configuration
snapshot also records the resolved non-secret image repository, registry
location and username, target host, SSH user/port, WireGuard port binding,
runtime sizing, builder, and asset path. The registry password remains a Kamal
secret reference and is never copied into the release record. Rollbacks restore
the recorded configuration and asset release as a unit. A changed host, SSH
user, or SSH port is rejected before the current service is stopped.

## Network and deployment

Set `KAMAL_HOST` to the dedicated host, `KAMAL_WIREGUARD_IP` to its WireGuard
address, `MANCALA_AI_ASSET_RELEASE` to the selected versioned asset directory,
`KAMAL_IMAGE` to `useruby/mancala-ai`, registry server/username/password, and
configure the secret in `.kamal/secrets`. Confirm WireGuard routing from both
Rails environments and approved monitoring hosts; verify an unapproved host is
blocked at Docker forwarding as well as host firewall rules.

```sh
bundle install
bundle exec kamal config -d mancala-ai
deploy/deploy.sh
```

The wrapper takes an exclusive lock, performs remote asset and target preflight,
stops the current container, deploys with the pinned Kamal CLI, then checks
readiness on `http://<WireGuard-address>:80`. Container health checks remain on
port 8080. Because
host port 80 cannot be bound concurrently, each update includes an expected API
interruption from stop through replacement health verification. Do not run
Kamal deploy/rollback directly; serialize all operations through the wrapper
from the designated deployment runner/workspace. Its `flock` and release-state
file are local to that workspace; restrict deployment access so operators do not
run independent wrappers from separate workspaces. Kamal's remote lock also
protects each individual CLI operation.

Check `GET /health/live`, `/health/ready`, legal move and terminal responses,
invalid JSON behavior, resource use, worker capacity and startup duration.

## Failure recovery and rollback

If replacement startup, health verification, or release-state persistence
fails, the wrapper stops the replacement to release host port 80, then invokes
`kamal rollback PREVIOUS_VERSION` with the previous asset mount restored and
checks readiness. Recovery failures are reported explicitly. Verify its reported model/profile identities before restoring
client traffic expectations. The wrapper retains the current and prior complete
release records; use its rollback subcommand for an intentional rollback:

```sh
deploy/deploy.sh rollback
```

Never change assets underneath a running container or pair an image with a
different service config/assets. Worker replacement within a running container
continues to use the mounted immutable release.
