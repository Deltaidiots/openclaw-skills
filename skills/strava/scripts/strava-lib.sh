#!/usr/bin/env bash
set -euo pipefail

# Shared helpers for Strava scripts.
# Secrets are loaded from ~/.config/strava.env by default.

STRAVA_ENV_FILE_DEFAULT="$HOME/.config/strava.env"

strava_load_env() {
  local env_file="${STRAVA_ENV_FILE:-$STRAVA_ENV_FILE_DEFAULT}"
  if [ ! -f "$env_file" ]; then
    echo "Missing env file: $env_file" >&2
    echo "Create it with STRAVA_CLIENT_ID/STRAVA_CLIENT_SECRET/STRAVA_REFRESH_TOKEN and chmod 600." >&2
    return 2
  fi

  # shellcheck disable=SC1090
  set -a
  . "$env_file"
  set +a

  : "${STRAVA_CLIENT_ID:?Missing STRAVA_CLIENT_ID}"
  : "${STRAVA_CLIENT_SECRET:?Missing STRAVA_CLIENT_SECRET}"
  : "${STRAVA_REFRESH_TOKEN:?Missing STRAVA_REFRESH_TOKEN}"
}

strava_token_refresh_json() {
  curl -s -X POST https://www.strava.com/oauth/token \
    -d client_id="$STRAVA_CLIENT_ID" \
    -d client_secret="$STRAVA_CLIENT_SECRET" \
    -d refresh_token="$STRAVA_REFRESH_TOKEN" \
    -d grant_type=refresh_token
}

strava_access_token() {
  # Extract access_token without jq.
  strava_token_refresh_json | sed -n 's/.*"access_token":"\([^"]*\)".*/\1/p'
}
