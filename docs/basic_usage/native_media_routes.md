# Native Media Routes

Pipelines whose generation stage runs native SGLang `multimodal_gen`, such as Cosmos3, serve the native SGLang image, video and action routes from the Omni server. These requests go straight to the native scheduler that the generation stage owns, so their request and response formats are those of native SGLang.

| Routes | Purpose |
| --- | --- |
| `POST /v1/images/generations`, `POST /v1/images/edits`, `GET /v1/images/{id}/content` | Image generation and editing |
| `POST /v1/videos`, `GET /v1/videos`, `GET /v1/videos/{id}`, `GET /v1/videos/{id}/content`, `DELETE /v1/videos/{id}` | Video jobs |
| `POST /v1/actions/generations`, `GET /v1/actions/metadata` | Action generation, when the checkpoint supports it |

```bash
sgl-omni serve --model-path /path/to/Cosmos3-Nano --port 8000
curl http://127.0.0.1:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "A red cube on a wooden table", "size": "640x640", "response_format": "b64_json"}'
```

## Behavior

- Only the routes above are served. Native weights, memory, LoRA, rollout, mesh, realtime, server info and stats routes are not, so use the Omni admin routes, which require the admin key when one is set.
- The native scheduler listens on `127.0.0.1` whatever host the server binds. Startup fails if the generation stage's `server_args_overrides` set another `host`.
- The routes start on the first request to them, which loads native SGLang's route modules into the server process once. That request takes several seconds longer, about 7 seconds with Cosmos3 Nano, and the server process keeps about 0.5 GiB more memory from then on. Requests that arrive meanwhile wait for the same start. A server whose clients never call these routes starts and runs as if they were absent.
- If the routes cannot start, for example because the prompt enhancer config cannot be read, these routes answer 503 with the cause until the server restarts, and the other routes keep serving.
- If the native scheduler or the generation stage stops, requests in flight and new requests on these routes get 503 at once, and the server stops when the generation stage fails.

## Server warmup

With `warmup_mode: server` in the generation stage's `server_args_overrides`, the generation stage runs native SGLang's server warmup before it reports ready, with the requests native SGLang's own server sends, so `/health` answers 200 only after it and the first request finds the generation stage warm. The native routes still start on their first request. As in native SGLang, a warmup that fails without explicit `warmup_resolutions` is logged as a warning and startup continues. The warmup input image is removed afterwards. The warmup is off by default. With Cosmos3 Nano on one H200 it costs:

| Cost against warmup off | Typical | At most |
| --- | --- | --- |
| Time until `/health` answers 200 | About +16 s (measured 95% interval +13 s to +19 s), 11 s of them native's automatic residency probe | +20 s |
| Energy of the generation GPU until ready | +9 kJ | +12 kJ |
| Energy of each other visible GPU until ready, idle during the longer startup | +1.3 kJ | +2.5 kJ |
| GPU memory in use from readiness until the first request | +5.4 GiB | +5.5 GiB |
| Host memory at readiness, all of it in the native worker | +0.9 GiB | +1.5 GiB |
| Host memory after the first request, all of it in the native worker | +0.1 GiB | +0.5 GiB |

In return the first request after readiness runs about 1.7 s faster.

## Media policy

`--allowed-local-media-path` and `--allowed-media-domain` apply to every media path and URL in a native route request, including native extra parameters and action observations. With `--allowed-local-media-path`, local paths and file URLs must resolve inside that directory, including the values native SGLang decodes from JSON encoded strings. Native SGLang opens a file URL as written, as a path relative to the server working directory, so send local media as plain paths. While `--allowed-media-domain` is set, remote URLs are refused on these routes, because native SGLang fetches them itself and follows redirects without checking each one, so send such media as an upload or a data URL. Without either option, media references are read as on other routes. Requests can never set server paths such as `output_path`, `perf_dump_path` or model paths.

## Retention

Generated images and videos stay in the generation stage output directory (`stages.generation.factory.output_dir`, `outputs` by default), and native SGLang keeps one job record per request in server memory until the server stops. Native SGLang has no expiry for either. `DELETE /v1/videos/{id}` only removes the job record, so it neither stops a running job nor deletes its file. Remove old files from the output directory on a long running server. Uploaded inputs are kept only for their request or video job.
