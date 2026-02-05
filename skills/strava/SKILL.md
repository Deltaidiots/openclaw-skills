---
name: strava
description: "Fetch Strava activities via OAuth (refresh token), summarize training, and troubleshoot scopes (activity:read_all)."
metadata:
  {
    "openclaw": {
      "emoji": "🏃",
      "requires": { "bins": ["curl"] }
    }
  }
---

# Strava

Use this skill to set up Strava OAuth for automation and fetch activity data (e.g., squash, runs, rides) in a scriptable way.

This skill is intentionally simple and auditable:

- Uses Strava’s official OAuth + REST API.
- Stores **no secrets** in the repo.
- Uses a local env file (chmod 600) for `STRAVA_CLIENT_ID/SECRET/REFRESH_TOKEN`.

## Quick start

### 1) Create a Strava API application

- Open: https://www.strava.com/settings/api
- Create an app.
- Set **Authorization Callback Domain** to: `localhost`

You’ll get:
- Client ID
- Client Secret

### 2) Create env file (recommended)

Create a user-only env file:

- `~/.config/strava.env` (chmod 600)

Contents:

- `STRAVA_CLIENT_ID=...`
- `STRAVA_CLIENT_SECRET=...`
- `STRAVA_REFRESH_TOKEN=...`

### 3) Authorize + exchange code

- Print the authorize URL:
  - `./scripts/strava-auth-url`

- Open it, click Authorize.

- Copy `code=...` from the redirect URL and exchange it:
  - `./scripts/strava-exchange-code`

Then copy `refresh_token` into `~/.config/strava.env`.

### 4) Test activities

- `./scripts/strava-activities --per-page 10`

## Common problems

### Activities endpoint returns `activity:read_permission missing`

Your authorization didn’t include activity scope.

- Re-authorize using the auth URL script.
- Confirm the redirect URL includes:
  - `scope=read,activity:read_all`

Also check Strava: https://www.strava.com/settings/apps

### Activities list is empty (`[]`)

Common causes:

- You authorized the wrong Strava account.
- Your Garmin → Strava sync hasn’t pushed activities yet.
- Privacy settings: verify the activity isn’t restricted in a way that blocks API access.

## Scripts

- `scripts/strava-auth-url` — prints an authorize URL.
- `scripts/strava-exchange-code` — exchanges a `code` for tokens (prints JSON).
- `scripts/strava-access-token` — exchanges refresh token for access token.
- `scripts/strava-activities` — fetches recent activities.
- `scripts/strava-summary` — summarizes the last N activities (plain JSON in/out).

## References

- See `references/strava-api.md` for endpoint notes and fields.
