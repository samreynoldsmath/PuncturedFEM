# Commit checklist

## Before commit
- [ ] update `TODO.md`
- [ ] update `CHANGELOG.md`
- [ ] run `isort`
  - [ ] source code
- [ ] run `black` / `flake8` / `mypy`
  - [ ] source code
  - [ ] tests
  - [ ] examples
- [ ] all tests pass: `pytest -v test`
- [ ] convert Juptyer notebooks: `./devtools/convert_examples.sh`
- [ ] all examples work

## Release
- [ ] merge to `main` branch
- [ ] update tag: `git tag -a vX.X.X -m "description"`
- [ ] push to GitHub: `git push`
- [ ] push to GitHub with tags: `git push --tags`
- [ ] GitHub: create release