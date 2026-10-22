# AI move API v1

The machine-readable contract is [`api-v1.json`](api-v1.json); its embedded schemas are the authoritative field definitions. Contract tests validate examples with the maintained `jsonschema` validator. JSON Schema considers an integral-valued number such as `1.0` an integer; the service's decoded Python boundary additionally requires strict `int` values (and rejects booleans, strings, and floats). Integers are bounded to exact IEEE-754-safe JSON integers where applicable (revision, durations and simulation counts ≤ 2^53−1); seeds/stores/pits are 0..48 and seed ≤ 2^32−1. Objects reject unknown properties.

## Rules and orientation

Rules are repository `kalah_v1`, 2 players, 6 pits per player, 4 seeds per pit initially, 48 seeds total. `player0_pits[0..5]` and `player0_store` are always player 0; player 1 fields are always player 1, irrespective of whose turn it is. Pit indices in a move result are actor-relative: 0..5 maps to the current player's `pits` array. Legal moves are nonempty current-player pits. Turns continue when the final seed lands in the actor's store; otherwise play changes player. Captures and terminal sweeps follow the repository rules. Rules end when the current player has no legal pits, then sweep remaining seeds; canonical terminal states therefore have all pits empty and settled stores. Their response result is `{ "kind": "terminal", "move": null }`. Nonterminal state must conserve exactly 48 seeds and have a legal move for the current player. The service must validate these semantic conditions in addition to JSON Schema validation.

## Request, response, errors

`POST /v1/move` accepts a caller-generated `request_id`, monotonic-per-game integer `game_revision`, complete absolute `state`, allowlisted `profile` identifier, and unsigned 32-bit `seed`. Profile is server-controlled and versioned (for example `balanced@v1`); a resolved profile identity is immutable/content-addressed and returned in `search_profile_id`. No client paths or search options are accepted. A model identity is immutable/content-addressed; profiles without a learned model report an explicit versioned no-model identity.

Success echoes `request_id` and `game_revision`, binds the result to `state_sha256`, and reports model/profile identities, nonnegative `elapsed_ms`, `completed_simulations`, and `termination_reason` (`completed` or `terminal`). A successful move must be legal for the exact submitted state. Search deadline expiration is an error and never returns an unchecked partial move.

All errors use `{ "error": { "code", "message", "request_id", "retryable" } }`. Status mapping: 400 malformed JSON/field types; 409 reused request ID with different payload; 422 invalid game state or unknown profile; 429 capacity exhausted; 503 no initialized usable worker; 504 search deadline exceeded; 500 internal failure. Errors are not successful move results. A repeated request ID with the identical payload while its result is available returns the same result; a different payload conflicts. Deduplication is service-instance scoped and is not durable exactly-once execution across restart. Clients must tolerate uncertain completion after connection loss.

`error.request_id` is required and nullable: the HTTP adapter returns `null` when a malformed request has no trustworthy identifier. This matches the OpenAPI schema and strict runtime `ErrorBody` model.

## In-process Python engine

`ml.alphazero_lite.serving.MoveEngine` accepts an explicit asset root, immutable `RuntimeBundle` definitions, and an allowlist of `SearchProfile` definitions. Initialize it once at process startup; initialization loads and hashes each artifact once, and requests cannot supply paths or search settings. Bundle kinds are `single` and `composed`; a composed bundle selects policy/value output sources while each source retains its own trunk. Keep test-only profiles separate from deployment profile registries. Missing or unsupported artifacts fail initialization.

Direct invocation:

```python
from ml.alphazero_lite.serving import MoveEngine, RuntimeBundle, SearchProfile, SearchSettings

engine = MoveEngine(
    asset_root="/srv/mancala/assets",
    bundles={"production": RuntimeBundle("production-v1", "single", ("incumbent",))},
    profiles={"balanced@v1": SearchProfile("balanced@v1", "balanced-v1", "production", SearchSettings(simulations=256))},
)
response = engine.select_move(request_dict)
```

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
