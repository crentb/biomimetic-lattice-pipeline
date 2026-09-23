# Security policy

## Supported versions

Security fixes are made on `main` and ship in the next release. The latest release on [PyPI](https://pypi.org/project/biomimetic-lattice-pipeline/) is supported.

| Version | Supported |
|---------|-----------|
| 0.2.x   | Yes       |
| < 0.2   | No        |

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub: open this repository's **Security** tab and select **Report a vulnerability**. Please do not open a public issue for a security report.

Each report is acknowledged and triaged. A confirmed vulnerability is fixed on `main`, released, and recorded in [CHANGELOG.md](CHANGELOG.md) once a fixed version is available.

## Automated checks

Every change must pass the continuous-integration security gate before it merges: gitleaks secret detection over the full history, bandit static analysis, pip-audit against known-vulnerable dependencies, and a trivy scan of the container image. The same gate re-runs weekly on `main`, and Dependabot vulnerability alerts are enabled.
