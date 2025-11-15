# Security Policy

## Supported Versions

We actively maintain and provide security updates for the following versions:

| Version         | Supported | Status              |
| --------------- | --------- | ------------------- |
| 1.5.1 (current) | ✅ Yes    | Active development  |
| 1.5.0           | ✅ Yes    | Security fixes only |
| < 1.5.0         | ❌ No     | Please upgrade      |

**Upstream**: This is a fork of [facebookresearch/BenchMARL](https://github.com/facebookresearch/BenchMARL). Critical
security issues should also be reported to upstream maintainers.

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in BenchMARL, please report it responsibly.

### Reporting Process

1. **DO NOT** open a public GitHub issue for security vulnerabilities
2. **Email** the maintainer directly: rechtevan (Evan Montgomery-Recht)
3. **Include** in your report:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Vulnerability Assessment**: Within 1 week
- **Fix Development**: Depends on severity
  - **Critical**: Within 7 days
  - **High**: Within 30 days
  - **Medium/Low**: Next release cycle

### Disclosure Policy

We follow responsible disclosure:

1. **Private disclosure**: Report received and confirmed
2. **Fix development**: Patch created and tested
3. **Coordinated disclosure**:
   - Notify upstream (facebookresearch/BenchMARL) if applicable
   - Allow time for coordinated release
4. **Public disclosure**: After fix is released and deployed

## Security Measures

### Current Security Infrastructure

- ✅ **CodeQL Static Analysis**: Automated security scanning via GitHub Actions
- ✅ **Dependency Scanning**: Regular pip-audit scans for vulnerable dependencies
- ✅ **Dependabot**: Automated dependency update PRs (weekly)
- ✅ **Type Safety**: MyPy type checking (reduces runtime errors)
- ✅ **Linting**: Ruff checks for code quality issues
- ✅ **Comprehensive Testing**: 92.74% test coverage

### Security Scanning Schedule

| Tool       | Frequency                     | Auto-remediation |
| ---------- | ----------------------------- | ---------------- |
| CodeQL     | Weekly (Mondays) + on push/PR | Manual review    |
| pip-audit  | Monthly                       | Manual review    |
| Dependabot | Weekly (Mondays)              | Auto-PR creation |

### Vulnerability Database Sources

- **PyPI Advisory Database**: Via pip-audit
- **GitHub Security Advisories**: Via Dependabot
- **CVE Database**: Via CodeQL
- **NPM Advisories**: N/A (Python-only project)

## Dependency Security

### Dependency Approval Policy

**Runtime dependencies must use**:

- MIT License
- Apache 2.0 License
- BSD License

**Development dependencies** can use any OSI-approved open source license.

### Known Dependencies

All runtime dependencies are audited and documented in:

- `.local/analysis/security_dependency_report.md`
- `.local/analysis/pinned_requirements.txt`

**Latest Audit**: 2025-11-15 **Vulnerabilities Found**: 0

### Dependency Updates

- **Security patches**: Applied immediately upon Dependabot notification
- **Minor updates**: Reviewed and merged weekly
- **Major updates**: Require manual testing and review

## Security Best Practices

### For Contributors

When contributing code:

1. **No Secrets**: Never commit API keys, passwords, or tokens
2. **Input Validation**: Validate all user inputs
3. **Safe Deserialization**: Avoid pickle with untrusted data
4. **Command Injection**: Sanitize inputs to system commands
5. **Path Traversal**: Validate file paths
6. **SQL Injection**: Use parameterized queries (if applicable)

### For Users

When using BenchMARL:

1. **Trusted Configs**: Only load configuration files from trusted sources
2. **Checkpoint Security**: Verify checkpoint file integrity before loading
3. **Environment Isolation**: Use virtual environments
4. **Version Pinning**: Pin dependencies for reproducible builds
5. **Regular Updates**: Keep BenchMARL and dependencies up-to-date

## Security Audit History

| Date       | Type          | Findings               | Action Taken              |
| ---------- | ------------- | ---------------------- | ------------------------- |
| 2025-11-15 | pip-audit     | 0 vulnerabilities      | ✅ Clean                  |
| 2025-11-15 | CodeQL        | Workflow configured    | ✅ Operational            |
| 2025-11-15 | Manual review | JSON serialization bug | ✅ Fixed (commit f9e6b3c) |

## Known Security Considerations

### Machine Learning Security

BenchMARL is a machine learning research library. Users should be aware of:

1. **Model Poisoning**: Train only on trusted data
2. **Adversarial Examples**: Models may be vulnerable to adversarial inputs
3. **Privacy**: Be cautious with sensitive data in training
4. **Resource Exhaustion**: Large models can consume significant resources

### Checkpoint Loading

Checkpoints use PyTorch's serialization:

- Checkpoints can execute arbitrary code when loaded
- **Only load checkpoints from trusted sources**
- Consider checkpoint integrity verification

### Configuration Loading

Hydra configuration system:

- YAML configs can reference environment variables
- **Only load configs from trusted sources**
- Validate config values before use

## Security Contact

**Maintainer**: rechtevan (Evan Montgomery-Recht) **Fork Repository**: https://github.com/rechtevan/BenchMARL
**Upstream**: https://github.com/facebookresearch/BenchMARL

For upstream security issues, contact Meta Platforms security team per their responsible disclosure process.

## Attribution

This security policy was created as part of the code quality infrastructure improvements tracked in Issue #1.

## Updates

This security policy is reviewed quarterly and updated as needed.

**Last Updated**: 2025-11-15 **Next Review**: 2026-02-15

______________________________________________________________________

**Related Documentation**:

- Security audit: `.local/analysis/security_dependency_report.md`
- Coverage analysis: `.local/analysis/coverage_analysis.md`
- Development guide: `CLAUDE.md`
