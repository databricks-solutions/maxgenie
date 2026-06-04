# Public Release Checklist

Use this checklist before requesting that the repository be created, made public, or annually recertified.

## Content Scope

- Product runtime, workspace skill files, templates, docs, and tests are in scope.
- Local workspace exports, benchmark payloads, old run artifacts, local reports, transcripts, and private planning files are out of scope.
- Public examples and tests must use synthetic fixture data only. Domain names, product names, and SQL snippets in tests are illustrative fixtures, not customer data.
- The source Genie space must remain untouched; MaxGenie applies candidate changes only to a managed clone.

## Security

- Confirm no credentials, tokens, passwords, profile files, or `.env` values are tracked.
- Confirm generated folders are covered by `.gitignore` and by the product-scope tests.
- Confirm no customer data, PII, private workspace exports, benchmark payloads, or query-history payloads are tracked.
- If a secret or private artifact is found, stop release work, remove it from the publication branch, and follow the appropriate remediation path.

## Legal And Review

- Select the approved repository license before making the repository public.
- Complete third-party dependency license review for the package dependencies declared in `pyproject.toml`.
- Confirm third-party code or assets are acknowledged with their applicable license before publication.
- Ensure at least one other repository owner or subject matter expert has reviewed the public-facing content.
- Confirm repository owners are assigned and understand the annual review and remediation responsibilities.
