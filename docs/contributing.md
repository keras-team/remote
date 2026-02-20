# How to contribute

We'd love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our community guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Contribution process

### Development setup

1. Install the package with development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

2. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

### Code quality and testing

Before submitting a pull request, please ensure your changes pass linting and unit tests.

- **Linting:** We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Run it with:
  ```bash
  ruff check .
  ```
- **Testing:** We use [Pytest](https://docs.pytest.org/) for unit tests. Run them with:
  ```bash
  pytest
  ```

### Submitting changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.
