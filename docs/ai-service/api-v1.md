# AI move API v1

The machine-readable contract is [`api-v1.json`](api-v1.json); its embedded schemas are the authoritative field definitions. Contract tests validate examples with the maintained `jsonschema` validator. JSON Schema considers an integral-valued number such as `1.0` an integer; the service's decoded Python boundary additionally requires strict `int` values (and rejects booleans, strings, and floats). Integers are bounded to exact IEEE-754-safe JSON integers where applicable (revision, durations and simulation counts ≤ 2^53−1); seeds/stores/pits are 0..48 and seed ≤ 2^32−1. Objects reject unknown properties.

## Rules and orientation

Rules are repository `kalah_v1`, 2 players, 6 pits per player, 4 seeds per pit initially, 48 seeds total. `player0_pits[0..5]` and `player0_store` are always player 0; player 1 fields are always player 1, irrespective of whose turn it is. Pit indices in a move result are actor-relative: 0..5 maps to the current player's `pits` array. Legal moves are nonempty current-player pits. Turns continue when the final seed lands in the actor's store; otherwise play changes player. Captures and terminal sweeps follow the repository rules. Rules end when the current player has no legal pits, then sweep remaining seeds; canonical terminal states therefore have all pits empty and settled stores. Their response result is `{ "kind": "terminal", "move": null }`. Nonterminal state must conserve exactly 48 seeds and have a legal move for the current player. The service must validate these semantic conditions in addition to JSON Schema validation.

## Request, response, errors

`POST /v1/move` accepts a caller-generated `request_id`, monotonic-per-game integer `game_revision`, complete absolute `state`, allowlisted `profile` identifier, and unsigned 32-bit `seed`. Profile is server-controlled and versioned (for example `balanced@v1`); a resolved profile identity is immutable/content-addressed and returned in `search_profile_id`. No client paths or search options are accepted. A model identity is immutable/content-addressed; profiles without a learned model report an explicit versioned no-model identity.

Success echoes `request_id` and `game_revision`, binds the result to `state_sha256`, and reports model/profile identities, nonnegative `elapsed_ms`, `completed_simulations`, and `termination_reason` (`completed` or `terminal`). A successful move must be legal for the exact submitted state. Search deadline expiration is an error and never returns an unchecked partial move.

All errors use `{ "error": { "code", "message", "request_id", "retryable" } }`. Status mapping: 400 malformed JSON/field types; 409 reused request ID with different payload; 422 invalid game state or unknown profile; 429 capacity exhausted; 503 no initialized usable worker; 504 search deadline exceeded; 500 internal failure. Errors are not successful move results. A repeated request ID with the identical payload while its result is available returns the same result; a different payload conflicts. Deduplication is service-instance scoped and is not durable exactly-once execution across restart. Clients must tolerate uncertain completion after connection loss.

`error.request_id` is required and nullable: the HTTP adapter returns `null` when a malformed request has no trustworthy identifier. This matches the OpenAPI schema and strict runtime `ErrorBody` model.

## HTTP application

`ml.alphazero_lite.serving.http.create_app(ServiceConfig(...))` owns the supervised pool for its ASGI lifespan. Supply trusted `asset_root`, immutable `RuntimeBundle` values with release-manifest SHA-256 identities, and the complete server-side `SearchProfile` allowlist; also configure `worker_count`, `startup_timeout`, `execution_timeout`, and `shutdown_timeout`. Requests can select only an allowlisted profile and cannot change paths, bundles, worker counts, or deadlines. Install the numerical dependencies from `requirements.txt` and `requirements-serving.txt`.

The entry point reads a trusted JSON configuration path from `MANCALA_AI_CONFIG`. It contains `asset_root`, `worker_count`, `startup_timeout`, `execution_timeout`, `shutdown_timeout`, `bundles` (mapping keys to `RuntimeBundle` fields), and `profiles` (mapping allowlisted names to `SearchProfile` fields, with nested `SearchSettings`). Bundle artifact paths are relative to the asset root and expected hashes must be supplied by the release manifest. Run one Uvicorn process initially, for example `MANCALA_AI_CONFIG=/etc/mancala-ai/config.json uvicorn ml.alphazero_lite.service:app --host 127.0.0.1 --port 8000`. Each additional API process creates another independent search-worker pool and multiplies configured capacity/resource use. Bind/deploy access through the approved WireGuard network; do not derive identities from mutable artifacts at startup.

Identical request IDs and payloads share in-flight computation and retained responses. Completed responses are bounded by configured maximum entry count and retention seconds (defaults: 1024 and 300 seconds); active entries are never evicted. State is process-local and lost on restart, so this is bounded duplicate suppression, not durable exactly-once execution. No automatic search retries occur. A disconnected caller relinquishes its waiter; worker cancellation is requested only after the last waiter leaves.

## In-process Python engine

`ml.alphazero_lite.serving.MoveEngine` accepts an explicit asset root, immutable `RuntimeBundle` definitions, and an allowlist of `SearchProfile` definitions. Initialize it once at process startup and call `close()` at shutdown (or use it as a context manager). Initialization authenticates artifacts against manifest identities; hashing whatever files happen to be present is not authentication. Each component's mandatory `expected_artifact_sha256` is the SHA-256 returned by `serving_artifact_identity(metadata.json, weights.json)`. The identity uses the fixed domain `azlite-serving-artifact-v1`, each fixed filename, its 8-byte big-endian length, and its complete bytes. Exporter metadata's `artifacts.weights_json_sha256` authenticates `weights.json`; `artifacts.weights_sha256` authenticates `model.npz` and is checked only when that checkpoint is included. JSON inference does not load the checkpoint: when it is absent, its digest is provenance only, while the manifest identity and JSON declaration authenticate the metadata and inference weights. For compatibility, metadata containing only `weights_sha256` is accepted as the legacy convention where that field authenticates `weights.json`; this legacy interpretation is not used when `weights_json_sha256` is present. Declared malformed or mismatching hashes are rejected. Runtime sidecars require their own explicit expected SHA-256 in `expected_policy_sha256`.

All paths are relative to `asset_root`, and resolved paths may not escape it. Runtime-policy paths in each sidecar are also relative to that root. A runtime-policy sidecar binds exact-root enablement, threshold, objective, tie rule, disabled exact-leaf mode, native implementation identity, and hashes for the executable probe and materialized tablebase. The executable must be a file with execute permission; the tablebase and probe must be real files, not directories or Git LFS pointer text. Architecture/encoding, output shapes, six finite nonnegative policy probabilities summing to one (float32 tolerance), and a finite value in [-1, 1] are checked before engine startup completes; outputs are rejected rather than repaired. Composed bundles must provide the same validated policy on both components. Their policy/value output routing is explicit (`current` or `candidate`) and contributes to the immutable bundle identity.

```python
from ml.alphazero_lite.serving import (
    MoveEngine, RuntimeBundle, SearchProfile, SearchSettings,
)

root = "/srv/mancala/assets"
bundle = RuntimeBundle(
    bundle_id="production-v1",
    kind="single",
    artifacts=("incumbent",),
    expected_artifact_sha256=("<release-manifest artifact SHA-256>",),
    # Omit only when the artifact intentionally has no search_policy.json.
    expected_policy_sha256=("<manifest SHA-256 of search_policy.json>",),
)
engine = MoveEngine(
    asset_root=root,
    bundles={"production": bundle},
    profiles={
        "balanced@v1": SearchProfile(
            "balanced@v1", "balanced-v1", "production",
            SearchSettings(simulations=256),
        )
    },
)
try:
    response = engine.select_move(request_dict)
finally:
    engine.close()
```

Use a release/build-produced serving manifest for expected hashes; do not calculate expected hashes from mutable assets at service startup. Bundle identities canonically bind component identities, composition routing, runtime policy semantics, and native asset hashes, but omit resolved absolute paths so identical bytes remain stable after relocation. Profile identities bind that bundle identity, the profile identifiers, and all search settings. Search settings and identifiers are validated during initialization. The old `deployment` profile flag was removed because it did not enforce a runtime behavior; `require_runtime_search_policy` is retained as an actual startup gate.

Terminal requests return without invoking the evaluator, search, or native probe. Exact-root adapter resources are initialized and warmed once per unique native asset identity in the engine, reused for requests at or below the configured threshold, and released on `close()`; initialization failures close resources already created. The serving engine dispatches eligible roots directly to the exact-root decision with its resolved threshold and native adapter. Outside the threshold normal arena PUCT runs through an evaluator wrapper that does not expose research artifact paths, so checkout-relative sidecar discovery is not used.

The engine raises typed failures for invalid requests, unknown profiles, runtime configuration errors, and invalid backend results. It is synchronous and does not itself enforce hard deadlines; a supervised worker layer must provide that guarantee.

`GET /health/live` returns 200 while the API process runs. `GET /health/ready` returns 200 only when search workers are initialized and usable, otherwise 503. Deployment must restrict access to approved WireGuard hosts; this contract defines no API token or OAuth scheme.

## State digest

Hash lowercase hexadecimal SHA-256 of UTF-8 bytes of the canonical state JSON object in precisely this key order: `player0_pits`, `player1_pits`, `player0_store`, `player1_store`, `current_player`. Encode compactly with no spaces, ASCII digits, decimal integer notation, no leading zeroes (except zero), no trailing newline, and no BOM. Arrays use comma separators and decimal integer elements. Example literal bytes (shown as UTF-8 text):

```text
{"player0_pits":[4,3,2,1,0,5],"player1_pits":[1,2,3,4,5,6],"player0_store":6,"player1_store":6,"current_player":0}
```

SHA-256: `a39102918d4cd3f2578c262ebd379dd69e0979c730ca1d15583584f54bf4ee9b`. Test vectors and request/response examples are in `fixtures/`.

## Rails caller responsibilities

Rails remains authoritative for game state and move application. Convert the selected actor-relative move index into Rails' move representation exactly once; compare echoed request/revision and state digest, reject stale or illegal results, then apply the move atomically. A stale response must never authorize applying a move to a newer game revision. On service errors, timeout, unavailable service, or invalid response, invoke the existing Ruby AI fallback. Example successful moves in fixtures are illustrative contract examples, not measured model predictions.
