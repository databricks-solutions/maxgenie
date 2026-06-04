# Security Policy

MaxGenie must not include credentials, tokens, private workspace exports, benchmark payloads, customer data, or other non-public information in source control.

## Reporting

Report suspected vulnerabilities, exposed credentials, or private data exposure through the appropriate Databricks security reporting channel. Do not disclose sensitive details in public issues, pull requests, or discussion threads.

## Repository Handling

- Keep local credentials in environment variables, local Databricks profile files, or ignored `.env` files.
- Do not commit generated run folders, benchmark payloads, query-history exports, workspace exports, transcripts, or local experiment reports.
- Rotate any credential immediately if it is suspected to have been committed, logged, or shared.
- Public examples and tests must use synthetic fixture data only.
