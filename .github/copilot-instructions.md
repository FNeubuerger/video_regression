# GitHub Copilot Instructions

## Code Style Guidelines

To ensure consistent and clean code, all Python code should conform to the [Black code style](https://black.readthedocs.io/en/stable/). Below are the instructions for using GitHub Copilot in this repository:

### General Guidelines
1. **Adhere to Black Formatting**: Ensure that all code suggestions align with Black's formatting rules.
2. **Line Length**: Limit all lines to 88 characters, as per Black's default configuration.
3. **String Quotes**: Use double quotes for strings unless single quotes are required for escaping.
4. **Imports**: Organize imports in compliance with [PEP 8](https://peps.python.org/pep-0008/), and avoid unused imports.

### Reviewing Copilot Suggestions
- Always review Copilot's suggestions to ensure they conform to Black's style.
- If a suggestion does not meet the style, manually adjust the code or regenerate the suggestion.

### Pre-Commit Hook
To enforce Black formatting, set up a pre-commit hook:
1. Install `pre-commit`:
    ```bash
    pip install pre-commit
    ```
2. Add the following configuration to `.pre-commit-config.yaml`:
    ```yaml
    repos:
      - repo: https://github.com/psf/black
         rev: stable
         hooks:
            - id: black
    ```
3. Install the pre-commit hook:
    ```bash
    pre-commit install
    ```

### Additional Tools
- Use `flake8` alongside Black to catch additional style issues.
- Configure your editor to auto-format code with Black on save.

By following these guidelines, we ensure a consistent and maintainable codebase.
