# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 1.0.x | Yes |

## Reporting a Vulnerability

If you discover a security vulnerability in StartupPulse AI, please report it responsibly. **Do not open a public GitHub issue for security vulnerabilities.**

### How to Report

1. Email **support@startuppulse.ai** with:
   - Description of the vulnerability
   - Steps to reproduce
   - Affected files or modules
   - Potential impact assessment
   - Suggested fix (if applicable)

2. You will receive an acknowledgment within **48 hours**.

3. A fix will be developed and released as a patch version once confirmed.

### What to Include

- **Type of vulnerability**: e.g., path traversal, dependency issue, data exposure
- **Affected component**: e.g., `src/model/predict.py`, dashboard, preprocessing pipeline
- **Reproduction steps**: minimal steps to demonstrate the issue
- **Severity estimate**: Low / Medium / High / Critical

## Security Considerations

### Data Handling

- The IBM HR Analytics dataset used in this project is a **public, synthetic dataset** from Kaggle. It contains no real employee PII.
- The prediction form accepts user-provided employee data. No data is persisted or transmitted outside the local application.
- Preprocessed splits (`data/processed/`) are generated locally and not served to external clients.

### Model Artifacts

- Trained model files (`.keras`, `.pkl`) are **not committed to version control** (excluded via `.gitignore`).
- Model files must be regenerated locally via `python -m src.model.train`.
- Do not load model files from untrusted sources.

### Dashboard

- The Streamlit dashboard runs locally by default (`localhost:8501`).
- Do not expose the dashboard to public networks without proper authentication and network controls.
- No authentication is built into the dashboard. If deploying publicly, place it behind a reverse proxy with auth.

### Dependencies

- Dependencies are pinned to version ranges in `requirements.txt`.
- Run `pip audit` periodically to check for known vulnerabilities in dependencies.
- TensorFlow, Scikit-learn, SHAP, Streamlit, and Plotly are the primary dependencies.

### Environment

- Never commit `.env` files, API keys, or credentials.
- The virtual environment (`.venv/`) is excluded from version control.
- Use a `.gitignore` to prevent accidental commits of sensitive files.

## Best Practices

| Practice | Details |
|---|---|
| Virtual environment | Always use `.venv` to isolate dependencies |
| No secrets in code | Never hardcode API keys, tokens, or passwords |
| Input validation | Validate employee data inputs before model prediction |
| Dependency updates | Run `pip audit` and update vulnerable packages |
| Network exposure | Do not expose the dashboard without authentication |
| Data minimization | Only process required employee features (30 fields) |

## Dependency Audit

```bash
# Check for known vulnerabilities
pip install pip-audit
pip-audit

# Update a vulnerable package
pip install --upgrade <package-name>
```

## Changes to This Policy

This security policy may be updated over time. Check this file for the latest version.
