# Demo runtime packaging

The runtime keeps the existing CPU PyTorch/MiniLM embedding model, Random Forest
engagement model, all nine recommendation artifacts, both JSON schemas, and the
eight Python modules imported by the serving entry point. Inference behavior and
the HTTP contract are unchanged.

`slim_runtime.py` removes dependency test directories, bytecode caches, and C++
development headers. It retains Python source, package metadata/licenses, runtime
`testing` helpers, model weights, and shared libraries. GNU `strip --strip-unneeded`
removes native debugging/unneeded symbols while retaining dynamic loader symbols.
Building C++ extensions inside the deployed runtime is intentionally unsupported.

The multi-stage Dockerfile copies cleaned packages into a fresh, pinned Python
base. Removing files in a later layer of the original image would not reclaim
the registry storage occupied by earlier layers. Binutils and its dependencies
stay in the intermediate packaging stage. Runtime bytecode writes are disabled;
imports can take longer without precompiled caches.

`requirements-prod.lock` records the dependency versions installed in the existing
verified image; CPU PyTorch is pinned separately in Dockerfile. The current
verification reused that dependency image, so it neither upgraded dependencies
nor downloaded models. Its packaging stage downloaded Debian build tools only.
The default dependency stage's fresh online rebuild was not exercised in this
cycle. Its MiniLM download still uses the upstream model's default revision;
verify the resulting model hashes before replacing the verified local image.

## Local verification on 2026-10-24

From the Python repository, the candidate was built with:

```powershell
docker build --pull=false --build-arg DEPENDENCIES_IMAGE=market-aura-python-verify:20261023-precheckpoint-d0d9d9 --tag market-aura-python-slim:20261024 .
python -B tests/test_slim_runtime.py
```

Candidate image: `sha256:3a384447c3e822d5448bdd1ef29de0d3c21573ed99e6fbc1b8f1a7744be4df66`.
The Node image remains unchanged:
`market-aura-node-verify:20261023-precheckpoint-d0d9d9`.

- All 68 asset/schema/runtime-source hashes and 55 package versions matched the
  original Python image. Eight reference-brand outputs matched exactly after
  removing generation timestamps. Invalid input handling also matched.
- Python regressions: 26 passed, one existing offline-dataset check skipped.
  Tests and existing pure-Python pytest tooling were mounted read-only, with
  network disabled, plugin autoload disabled, and no pytest cache writes.
- Backend Linux regressions: 366 passed with network disabled.
- The complete register/login/onboard/generate/persist/history journey passed
  against disposable, memory-backed MongoDB and the real Python service on an
  internal Docker network. Groq was intercepted locally. Carousel slides, ML
  grounding, persistence, owner isolation, pagination, and unauthorized history
  access were checked using existing journey assertions.
- Node health/readiness returned 200/200; a MongoDB outage produced 200/503,
  then readiness recovered. Python health returned 200; `/ready` remains 404.
- Two Python fresh-container starts with one CPU and 2 GiB memory reached health
  in 16.26 and 13.51 seconds. Local Node-to-Python requests took 58-104 ms after
  the full journey. Startup plus the sampled request remained below 60 seconds;
  the client timeout probe also passed. These are local Docker measurements,
  with warm host/image caches, single-thread BLAS settings, and no Cloud Run IAM
  or queue latency. Peak observed Python process RSS was approximately 762 MiB.
- All verification containers and the internal network were removed. Existing
  application containers and original images were preserved.

## Storage result and boundary

Unique exported image blobs were counted locally. Existing compressed blobs were
counted as stored; uncompressed blobs were measured with gzip level 6, including
small archive metadata. No images were uploaded.

| Image | Compressed estimate |
| --- | ---: |
| Unchanged Node | 86.608 MiB |
| Slim Python | 422.905 MiB |
| Combined including metadata | 509.518 MiB |

Combined bytes: 534,268,866. Against a 512 MiB allowance this leaves only
2.482 MiB. This narrowly fits one image pair; it does not establish a zero bill.
The actual registry representation and other storage on the billing account must
be checked before uploading. Additional versions, rollback images, or separate
costs can exceed the free allowances. Keep the original local images for rollback;
no upload, deployment, or broader optimization is authorized by this verification.

Detailed local evidence (logs, comparison inputs/results, harnesses, size report):
`C:\Users\singh\AppData\Local\Temp\market-aura-slim-0e348f53d7ac448885db38e1d4592a86`.
