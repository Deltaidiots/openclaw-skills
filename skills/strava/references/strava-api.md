# Strava API notes (for this skill)

Base: https://www.strava.com/api/v3

OAuth

- Authorize: `GET https://www.strava.com/oauth/authorize?...`
- Token exchange:
  - `POST https://www.strava.com/oauth/token` with `grant_type=authorization_code`
- Refresh:
  - `POST https://www.strava.com/oauth/token` with `grant_type=refresh_token`

Endpoints

- `GET /athlete`
- `GET /athlete/activities?per_page=...&after=...&before=...`
- `GET /athletes/{id}/stats`

Scopes

- For activities: `activity:read_all`.

Gotcha: The redirect URL may show `scope=` empty if the consent didn’t grant requested scopes.
