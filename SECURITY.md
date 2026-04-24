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
- **Configuration**: Controlled via `RATE_LIMIT_ENABLED` environment variable (default: `true`, enabled by default)
- **Health check exempt**: `/health` endpoint is not rate-limited
- **Trusted proxies**: Middleware respects `X-Forwarded-For` header via `TRUSTED_PROXY_COUNT` environment variable (default: 0). Set this correctly if behind a reverse proxy (e.g., load balancer).
- **Testing**: Set `RATE_LIMIT_ENABLED=false` via `monkeypatch.setenv()` in tests to disable rate limiting and prevent spurious failures

The rate limiter uses an in-memory store and is safe for single-instance deployments. For multi-instance production deployments, consider externalizing to Redis or Memcached.

## Prompt injection prevention

**Primary defense**: The `build_prompt()` function in `src/generation/prompt_builder.py` sanitizes all user input before building the prompt via the `_sanitize_for_prompt()` helper. This function:
- Normalizes Unicode via `unicodedata.normalize("NFKC")` to prevent homograph attacks
- Removes all control characters (Unicode category "C": Cc, Cf, Co, Cs, Cn) to strip newlines, tabs, and other whitespace-like characters that might break out of the prompt template
- Strips leading/trailing whitespace

This comprehensive sanitization prevents prompt injection attacks while preserving printable Unicode characters.

**Request validation**: The `QueryRequest` model in `src/api/models.py` enforces:
- `query`: required, length 1–2000 characters
- `entity_name`: optional, regex-validated (letters, digits, spaces, hyphens, underscores, apostrophes only)

The `parse_query()` function in `src/api/query_parser.py` further validates that the query is non-empty after stripping whitespace.

**Response sanitization**: Generated answers are not HTML-escaped by default. If deploying a web UI, ensure HTML entities in `answer` and other string fields are escaped on the client side.

## Logging

The `src/utils/logging.py` module configures centralized logging:

- **Root level**: Controlled via `LOG_LEVEL` environment variable (default: `INFO`)
- **httpx suppressed**: `httpx` is set to `WARNING` level to prevent per-request INFO logs from leaking internal Qdrant instance URLs or other sensitive connection details
- **Format**: Human-readable: `%(asctime)s %(levelname)-8s %(name)s — %(message)s`
- **Output**: Always to `stdout` for container/serverless compatibility
- **Re-entrant safe**: Multiple calls to `setup_logging()` are safe; handlers are not duplicated

Ensure logs are not persisted to disk in production without proper access controls, as they may contain PII or entity information from queries.

## Request validation and size limits

The API enforces request size and structure validation via middleware in `src/api/app.py`:

- **Body size limit**: `BodySizeLimitMiddleware` rejects requests with `Content-Length` exceeding 64 KB (far above any valid query payload)
- **Negative Content-Length guard**: Requests with negative `Content-Length` are rejected with 413 (Request Entity Too Large)
- **Invalid Content-Length header**: Non-numeric `Content-Length` headers are rejected with 400 (Bad Request)

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

## HTTP security headers

The API adds security headers to all responses via `SecurityHeadersMiddleware` in `src/api/app.py`:

- **X-Content-Type-Options**: `nosniff` — prevents MIME type sniffing
- **X-Frame-Options**: `DENY` — prevents clickjacking by forbidding iframes
- **Referrer-Policy**: `strict-origin-when-cross-origin` — limits referrer leakage
- **Content-Security-Policy**: `default-src 'none'` — restricts inline scripts and external resources
- **Strict-Transport-Security**: `max-age=31536000; includeSubDomains` — enforced only if `HTTPS_ENABLED=true`

## CORS and origin control

The API configures CORS via `CORSMiddleware` in `src/api/app.py`:

- **Default**: Allow all origins (`ALLOWED_ORIGINS="*"`) with credentials disabled
- **Custom origins**: Set `ALLOWED_ORIGINS` to a comma-separated list (e.g., `https://example.com, https://app.example.com`) to restrict and enable credentials
- **Methods**: Only `GET` and `POST` are allowed
- **Headers**: Only `Content-Type` and `Authorization` are allowed in requests

## Stats endpoint authorization

The `/stats` endpoint requires authentication if `STATS_API_KEY` environment variable is set:

- **Query format**: `GET /stats` with `Authorization: Bearer <STATS_API_KEY>` header
- **Validation**: Uses constant-time comparison (`hmac.compare_digest()`) to prevent timing attacks
- **Default**: If `STATS_API_KEY` is not set, the endpoint is public

## Supported versions

This project does not have versioned releases. Security fixes go to `main` directly.
