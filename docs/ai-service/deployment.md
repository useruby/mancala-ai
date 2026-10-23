# Python AI service deployment

The service image uses CPython 3.12.8 on Debian Bookworm, FastAPI/Uvicorn,
one Uvicorn process, and Pebble-owned search workers. Container dependencies are
locked in `deploy/requirements-container.txt`; tests are pinned in
`deploy/requirements-test.txt`. Numerical libraries are limited
to one thread per worker. The image runs as UID 10001. `SIGTERM` is delivered
to Uvicorn (exec-form entry point); Uvicorn may wait 25 seconds for requests,
then the worker pool may need up to 10 seconds to shut down. Kamal and Docker
each allow 45 seconds, including a 10 second margin.

## Assemble an immutable asset release

For local verification, install CPython 3.12.8 and Ruby 3.3.6, then use the
repository-local Python environment and the locked Ruby toolchain:

```sh
python3.12 -m venv .venv-deploy
.venv-deploy/bin/python -m pip install -r deploy/requirements-test.txt
bundle install
```

`Gemfile.lock` pins Kamal 2.5.2 and its Ruby dependencies. CI uses the same
Python/Ruby versions and a Linux runner with Docker enabled. Kamal config CI
uses the reserved documentation-only address `192.0.2.1` and does not connect
to a host or registry.

Create a release manifest outside the image containing each source path,
destination path, file kind, and SHA-256 from the release authority. Include
every model weight and metadata file, search-policy sidecar, executable native
probe, tablebase, and the trusted `service.json`. The asset tool rejects
missing files, LFS pointer text, incorrect hashes, unsafe/duplicate paths, and
non-executable or non-runnable probes. Do not compute trusted hashes from the
candidate directory being packaged. Keep the release manifest with the signed
release record and protect its provenance independently.

```sh
python deploy/assemble_release.py /secure/release/manifest.json \
  /srv/mancala-ai/releases/2026-10-23.1
```

The JSON service config must explicitly set `asset_root` to
`/releases/current`, exact bundle composition, artifact/policy expected hashes,
profile search settings, exact-root configuration, and `worker_count`. No
defaults in the packaging tool rewrite engine behavior. Mount the release
directory read-only; retain the engine's path-containment checks.

The checked-in CI fixture is release `ci-seed48-nextgen-s455-default-value-iter1`:
artifact identity `4551493a81efeeb83e84339724a0199cf7e00cb3b04e32475ab9e5ac07a3781f`,
policy SHA-256 `b13ced03deda6bd2d1737c6d339aececfe0b3730563ca800e36d12944b08fb8d`,
native probe SHA-256 `d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48`,
and tablebase SHA-256 `f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7`.
This authenticates the tracked model for verification only. For staging and
production, obtain explicit identities from the release authority and provide
a matching trusted manifest; never infer expected identities from candidate
files during assembly.

## Build and stage

Install the pinned Kamal CLI with `bundle install` and Docker. Set the
following operator values in the deployment shell or protected environment:

* `KAMAL_HOST`: dedicated staging server address/name.
* `KAMAL_WIREGUARD_IP`: that server's WireGuard interface address.
* `KAMAL_RELEASE_PATH`: absolute host path containing versioned releases and
  the `current` symlink.
* `KAMAL_IMAGE`: registry-relative repository path (for example
  `useruby/mancala-ai`, without registry hostname or tag).
* `KAMAL_REGISTRY_SERVER`, `KAMAL_REGISTRY_USERNAME`, and
  `KAMAL_REGISTRY_PASSWORD`.
* Configure `.kamal/secrets` with `KAMAL_REGISTRY_PASSWORD=$KAMAL_REGISTRY_PASSWORD`;
  keep the secret file out of source control.
* Docker-aware firewall allowlist source addresses for approved Rails,
  staging, and monitoring hosts; supply no public source range.

```sh
bundle exec kamal config -d staging
bundle exec kamal deploy -d staging
```

Kamal builds and pushes an image tagged with the clean source revision. Enable
immutable-tag enforcement in the registry so a revision cannot be overwritten;
do not deploy a dirty tree or reuse a version tag for different contents.

`config/deploy.yml` uses `ENV.fetch` for host-specific values so absent values
fail rather than target an invented host. Staging and production overlays are
separate. The service has no proxy/public listener: port 8080 binds only to the
configured host WireGuard address. Apply host firewall restrictions to
Docker's forwarding path as well as host INPUT rules, and confirm an approved
source can reach `/health/live` while an unapproved host cannot.

Before requesting staging acceptance, check `kamal app logs -d staging`,
`GET /health/live`, `GET /health/ready`, a legal move, terminal response, and
invalid JSON behavior from the approved network. Check memory, CPU saturation,
worker capacity/restarts, and startup duration with monitoring. Size worker
count from measured concurrent demand and available memory; each worker loads
its own model/tablebase resources. Container defaults are 2 CPUs and 2 GiB and
are adjustable through `KAMAL_CPUS` and `KAMAL_MEMORY`.

Production deployment follows successful staging acceptance and uses its own
host, protected registry credentials, and trusted asset release/configuration.
There is no rollout feature flag. For rollback, first move the release path's
`current` symlink to the previous compatible release (which contains its
matching service config), then run
`bundle exec kamal rollback <previous-revision> -d staging` (or production).
The revision selects the retained immutable image. Restore image, configuration,
and assets as one versioned unit; never pair an old image with different assets.

## Operator values still required for staging

Dedicated server hostname, WireGuard bind address, release mount path, registry
address and credentials, image namespace, immutable-tag registry policy, approved source IPs,
firewall implementation/rules, trusted release manifest and service config,
model/tablebase release identities, and chosen worker/CPU/memory sizing.
No hosts are provisioned or deployed by this change.
