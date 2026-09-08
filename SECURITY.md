# Security policy

## Supported versions

OpenMem is pre-1.0. Security fixes are applied to the latest `main` branch. Older commits are not guaranteed to receive fixes.

## Report a vulnerability

Please do not open a public issue for an undisclosed vulnerability. Use GitHub’s private vulnerability reporting for this repository, or contact the maintainers through the private contact listed in the repository profile.

Include:

- affected commit or version;
- impact and likely attack path;
- minimal reproduction steps;
- any suggested fix.

You should receive an acknowledgement within 7 days. Please allow time for investigation and a fix before public disclosure.

## Protecting data

Do not attach real database files, API keys, tokens, or user conversations to issues or pull requests. OpenMem redacts common secrets before event persistence, but redaction is a safety layer, not a substitute for data minimization.
