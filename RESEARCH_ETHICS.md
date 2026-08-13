# Research ethics

This document states the boundaries this project operates within. It is a commitment,
not a disclaimer.

## Phase 1 — this repository

**Read-only.** The harness queries the public PyPI JSON API and the public npm registry to
determine whether a package name exists. It performs no writes of any kind.

Specifically, Phase 1 does **not**:

- register, reserve, claim, or publish any package on any registry
- upload any artifact anywhere
- contain, generate, or distribute any payload, benign or otherwise
- attempt to acquire an existing package, namespace, or maintainer account

If code in this repository ever appears to publish to a package registry, that is a bug —
please open an issue.

## Being a good citizen of the registries

- All registry access is rate-limited and concurrency-capped.
- Results are cached on disk, so repeated runs re-query as little as possible.
- Requests identify themselves via a descriptive User-Agent pointing at this repository.
- A network error, timeout, or non-404 response is **never** recorded as "package does not
  exist". Only an explicit 404 counts as a negative result.

## Handling of findings

A list of hallucinated-but-unregistered package names is itself sensitive: publishing it
tells an attacker exactly which names are worth registering. Before any public write-up,
the following are decided deliberately rather than by default:

- whether specific names are published, or only aggregate rates and categories
- whether affected names are disclosed to PyPI and npm security contacts in advance
- how much notice registries are given before publication

The default posture is **coordinate first, publish second**.

## Phase 2 — not in this repository, and not yet started

A possible later phase involves *defensive* registration: claiming high-recurrence
hallucinated names so an attacker cannot. That phase is out of scope here and will not
begin without:

1. Prior coordination with the PyPI and npm security teams.
2. A written commitment that any registered package is a **benign no-op** — no install
   hooks, no network calls, no telemetry, no data collection of any kind. Note that even
   an "empty" package executes its build script on install; a no-op must be verified as
   such, not assumed.
3. A stated custody and hand-over plan for any name registered, so packages do not become
   orphaned or transferable to a bad actor later.
4. A public statement of what was registered and why.

Nothing in Phase 2 will involve a payload, a beacon, or any collection of data from
people who install a package. Measuring real-world installation rates is explicitly **not**
a justification for shipping code that phones home.

## Contact

Issues and disclosure: open an issue on this repository.
