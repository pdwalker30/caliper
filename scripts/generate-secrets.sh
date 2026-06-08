#!/usr/bin/env bash
# ============================================================================
# Caliper — generate strong secrets for .env
# ============================================================================
# Prints suggested values for every *_SECRET / *_KEY / *_SALT / *_AUTH variable
# in .env. Copy/paste into your .env file. Re-run anytime — values are random,
# never the same twice.
#
# Requires: openssl  (ships on macOS / Linux; on Windows use Git Bash / WSL)

set -euo pipefail

if ! command -v openssl >/dev/null 2>&1; then
  echo "ERROR: openssl is required but not found on PATH." >&2
  exit 1
fi

rand_hex_32() { openssl rand -hex 32; }
rand_b64_32() { openssl rand -base64 32 | tr -d '\n'; }

cat <<EOF
# --- Generated $(date -Iseconds) ------------------------------------------
# Paste these into your .env file (replacing the empty values).

# Langfuse
LANGFUSE_NEXTAUTH_SECRET=$(rand_b64_32)
LANGFUSE_SALT=$(rand_b64_32)
LANGFUSE_ENCRYPTION_KEY=$(rand_hex_32)

# Redis (Langfuse worker auths against this)
REDIS_AUTH=$(rand_hex_32)

# LiteLLM master key — must start with sk-
LITELLM_MASTER_KEY=sk-$(rand_hex_32)
EOF
