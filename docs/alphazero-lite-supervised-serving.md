# Supervised search workers

`ml.alphazero_lite.serving.workers.SearchWorkers` supervises Pebble processes for
serving search. Configure `worker_count`, `start_method` (`spawn` by default),
`startup_timeout`, `execution_timeout`, and `shutdown_timeout`. There is one
single-worker Pebble pool per capacity slot, so work is never queued behind an
unbounded shared pool queue.

Each worker runs `MoveEngine` construction (including artifact validation,
model loading, and configured native warm-up) in its process initializer. The
parent advertises the slot as ready only after a nonce-bound readiness task
returns the expected generation token and worker PID. `status.slots` reports
each slot's generation, `starting`/`ready`/`busy`/`unavailable` state, PID, and
startup/recovery error; `status.workers` is the number of verified idle workers.
After timeout, cancellation, or worker death, the slot is removed from usable
capacity. Pebble terminates/replaces a worker for timed-out, cancelled-running,
or dead tasks; the parent queues a new token handshake on that slot's pool and
only counts the slot again after that handshake completes. A failed or hung
replacement initializer leaves the slot unavailable and visible in status.
Generation checks discard late readiness callbacks.

`startup_timeout` bounds each initial or replacement initialization handshake;
when it expires, the slot pool is stopped and joined. `execution_timeout` is
Pebble's per-task computation deadline and terminates the executing worker.
`shutdown_timeout` bounds graceful draining and the forced-stop join. Search
requests are strictly validated and copied to a serializable Python snapshot;
results are serializable response dictionaries. A model is not loaded per
request.

Admission is immediate and bounded by the number of `ready` slots. If all slots
are busy, starting, or unavailable, `submit` raises a typed capacity/startup
failure rather than creating a waiting queue. This is not a durable job queue.
The returned concurrent future supports running cancellation through Pebble's
`ProcessFuture.cancel()`. When Pebble accepts cancellation, the task future
completes with `SearchCancelledError`; running work is terminated and capacity
returns only after a valid readiness handshake. A completed future cannot be
cancelled. Application exceptions are propagated without worker replacement;
execution deadlines and worker death produce `ExecutionTimeoutError` and
`WorkerCrashedError` respectively.

## Native child-process guarantee

Supervised execution currently requires Linux. The native probe is launched
through a small Python `exec` bootstrap that captures the worker PID, sets
`PR_SET_PDEATHSIG` to `SIGKILL`, and compares the actual parent PID before
`exec`-ing the native binary. This closes the race where the worker dies before
the child installs its parent-death signal. It does not depend on a worker
`finally` block. The pool constructor fails explicitly on platforms without
this guarantee. The guarantee covers the direct probe process used by the
current native adapter; it is not a general process-tree supervisor for
arbitrary future native tools.

During normal application lifespan teardown, stop admission and call
`shutdown(graceful=True)`. It drains until `shutdown_timeout`, then forcibly
stops remaining pools. `shutdown(graceful=False)` immediately uses Pebble's
forced stop path. For a future async HTTP adapter, await without blocking its
event loop:

```python
future = workers.submit(request)
response = await workers.await_result(future)
```

No HTTP mapping, public job identifiers, persistence, retries, or deployment
configuration are included.
