# Branch Protection Checklist (Security Gates)

Use this checklist in GitHub repository settings for `main` (and `master` if used).

## Required Status Checks

- `Python Security Gates`
- `Container Security Gates`

## Protection Rules

- Require pull request before merging
- Require at least 1 approving review
- Dismiss stale approvals when new commits are pushed
- Require status checks to pass before merging
- Require branches to be up to date before merging
- Restrict force pushes
- Restrict branch deletion

## Optional But Recommended

- Require conversation resolution before merging
- Require signed commits
- Restrict who can push to protected branch
- Enable merge queue for busy repos

## Security Operations

- Enable Dependabot security updates
- Enable secret scanning + push protection
- Enable code scanning alerts (SARIF uploads from Trivy workflow)
