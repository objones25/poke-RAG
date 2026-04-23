# SECURITY.md

## Reporting a vulnerability

This is a private/personal project. If you find a security issue, open a GitHub issue marked `[security]` or email the maintainer directly. Do not post vulnerability details in a public issue without first giving the maintainer a chance to patch it.

Expected response time: within 7 days.

## Secrets and credentials

This project handles several sensitive values that must never be committed:

| Secret           | Where it's used                                                      |
| ---------------- | -------------------------------------------------------------------- |
| `HF_TOKEN`       | HuggingFace model downloads (gated models)                           |
| `RUNPOD_API_KEY` | RunPod GPU provisioning                                              |
| `QDRANT_API_KEY` | Qdrant Cloud or hosted Qdrant instance (not needed for local Docker) |
| `QDRANT_URL`     | Qdrant host URL in non-local environments                            |

**Rules:**

- Store all secrets in a `.env` file at the project root
- `.env` is in `.gitignore` — confirm this before committing
- Load secrets via `python-dotenv` or environment variables — never hardcode them
- If a secret is accidentally committed, rotate it immediately and rewrite history (`git filter-repo`) or contact the service provider to revoke it

Example `.env` structure (safe to document here — do not fill in real values):

```env
HF_TOKEN=
RUNPOD_API_KEY=
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

Local Qdrant (Docker) does not require an API key. `QDRANT_API_KEY` is only needed for hosted/cloud instances.

**Secret handling in config**: The `qdrant_api_key` field in `src/config.py` is a `SecretStr` from Pydantic. It is never logged in plaintext; the value is masked in string representations and logs.

## Rate limiting

The API enforces rate limiting on all endpoints via `RateLimitMiddleware` in `src/api/app.py`:

- **Limit**: 20 requests per minute per IP address
- **Configuration**: Controlled via `RATE_LIMIT_ENABLED` environment variable (default: `true`)
- **Testing**: Set `RATE_LIMIT_ENABLED=false` via `monkeypatch.setenv()` in tests to disable rate limiting and prevent spurious failures
- **Header inspection**: Middleware inspects `X-Forwarded-For` header for reverse-proxy deployments

The rate limiter uses an in-memory store and is safe for single-instance deployments. For multi-instance production deployments, consider externalizing to Redis or Memcached.

## Prompt injection prevention

**Primary defense**: The `src/generation/prompt_builder.py` module strips newline characters (`\n`, `\r`, `\t`) from user-supplied query strings before building the prompt. This mitigates prompt injection attacks that attempt to break out of the prompt template via multi-line input.

**Request validation**: The `QueryRequest` model in `src/api/models.py` enforces:
- `query`: required, length 1–2000 characters
- `entity_name`: optional, regex-validated (letters, digits, spaces, hyphens, underscores, apostrophes only)

**Response sanitization**: Generated answers are not HTML-escaped by default. If deploying a web UI, ensure HTML entities in `answer` and other string fields are escaped on the client side.

## Logging

The `src/utils/logging.py` module configures centralized logging:

- **Root level**: Controlled via `LOG_LEVEL` environment variable (default: `INFO`)
- **httpx suppressed**: `httpx` is set to `WARNING` level to prevent per-request INFO logs from leaking internal Qdrant instance URLs or other sensitive connection details
- **Format**: Human-readable: `%(asctime)s %(levelname)-8s %(name)s — %(message)s`
- **Output**: Always to `stdout` for container/serverless compatibility

Ensure logs are not persisted to disk in production without proper access controls, as they may contain PII or entity information from queries.

## Dependencies

This project uses ML libraries (`transformers`, `torch`, `unsloth`, `trl`, `peft`) that pull in large dependency trees. Known practices:

- Pin all dependencies via `uv.lock` — do not use loose version ranges in production
- Run `uv lock --upgrade` periodically and review the diff before committing
- Check [HuggingFace security advisories](https://github.com/advisories?query=ecosystem%3Apip) for packages in this stack
- Do not install untrusted packages from community RunPod templates without reviewing them

## Model weights

- Model weights downloaded from HuggingFace are not audited for backdoors or adversarial modifications
- Only use weights from the official `google/` namespace on HuggingFace for Gemma models
- LoRA adapters trained on RunPod should be treated as untrusted if loaded from external sources

## Data files

`processed/` contains scraped public data (Bulbapedia, PokéAPI, Smogon). It does not contain PII. Do not add any user data, credentials, or private information to this directory.

## RunPod

- Use RunPod Secure Cloud instances if the training data or model weights are considered sensitive
- Community Cloud is acceptable for this project given no PII is involved
- Do not hardcode `RUNPOD_API_KEY` in any training script — load from environment
- Terminate pods when not in use to avoid unnecessary exposure

## Supported versions

This project does not have versioned releases. Security fixes go to `main` directly.
