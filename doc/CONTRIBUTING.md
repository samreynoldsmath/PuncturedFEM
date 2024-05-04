# Contributing Guide
This is a brief guide describing how to contribute to the `PuncturedFEM` project.

### TL;DR
First-time setup:

- Fork the GitHub repo
- Checkout a new branch
- Create and activate a virtual environment
- Install dev tools
- Install this package in edit mode

Make your changes:

- Use `numpy`-style docstrings
- Use `pylint` and `mypy` for linting
- Use `isort` and `black` (in that order) for formatting
- Run tests with `pytest`
- Clear all Jupyter notebook outputs

Final steps:

- Commit your changes
- Push to your fork
- Open a pull request

### Prerequisites
- [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) (don't forget to [configure your username and email](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup)
- [python3](https://www.python.org/) (version 3.9 or later)
- [pip](https://pypi.org/project/pip/) (should be installed with Python)
- [venv](https://docs.python.org/3/library/venv.html) (or another virtual environment manager)

## First-time setup

### Fork & clone
- [Fork the repo](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) on GitHub and make sure it is up-to-date
- Clone your fork to your local machine, and change directory to your local repo:
  ```bash
  git clone https://github.com/myusername/PuncturedFEM my/path/to/PuncturedFEM
  cd my/path/to/PuncturedFEM
  ```

### Set up a virtual environment
- Configure a virtual environment:
  ```bash
  python3 -m venv .
  ```
- Activate your virtual environment:
  - Linux/Mac
    ```
    source bin/activate
    ```
  - Windows command line
    ```bash
    venv\Scripts\activate.bat
    ```
  - Windows PowerShell
    ```bash
    venv\Scripts\Activate.ps1
    ```

### Install dependencies and developer tools
- Use `pip` to install developer tools:
  ```bash
  pip install -r requirements-dev.txt
  ```
- Install this package in edit mode:
  ```bash
  pip install -e .
  ```

## Make your changes
- Make sure that the `main` branch is up-to-date:
  ```
  git switch main
  git pull
  ```
- Checkout a new branch, named according to the contribution you wish to make (short but descriptive is best):
  ```bash
  git switch -c descriptive-branch-name
  ```
- Set the upstream branch:
  ```bash
  git push --set-upstream origin descriptive-branch-name
  ```
- Make your changes and save your files

### Best practices
- It is best practice to make small, incremental changes and make [commits](#commit-your-changes) as you go
- This way if you make a mistake, it is easy to [return to a version of your code that works](https://www.atlassian.com/git/tutorials/undoing-changes)
- Running [tests and linters](#tests-and-linters-for-source-code) frequently can help catch errors early
- Write and update documentation as you go

## Run formatters
This step helps make the code you've written more readable and standardized.

- Be sure that your virtual environment is active
- Format import statements with [isort](https://pypi.org/project/isort/) and format code with [black](https://pypi.org/project/black/):
  ```bash
  isort puncturedfem tests examples
  black puncturedfem tests examples
  ```
- **NOTE:** Always run `black` after running `isort`
- If contributing Jupyter notebooks, clear all outputs:
  ```bash
  jupyter nbconvert --clear-output --inplace examples/*.ipynb
  ```

## Tests and linters (for source code)
This step ensures that your code is correct and follows best practices.

### Run tests
- Be sure that your virtual environment is active
- Use [pytest](https://docs.pytest.org/en/8.0.x/) to run the tests in the `test/` directory
  ```bash
  pytest tests
  ```
- Optionally, you can use the `-s` flag to see print statements (helpful for debugging)
  ```bash
  pytest -s tests # show print statements
  ```
- Be sure all tests pass before proceeding

### Run linters
- Be sure that your virtual environment is active
- `pylint` will print the filename, line number, and a description of any errors it finds:
  ```bash
  pylint puncturedfem tests
  ```
- **Hint:** If you encounter formatting errors (e.g. lines over 80 characters, [running black](#run-formatters) may resolve some of them)
- Similarly for `mypy`:
  ```bash
  mypy puncturedfem tests
  ```
- Fix any errors before proceeding

## Write documentation (for source code)
This step ensures that users will be able to find information about how to use the new features you've added.

### Docstrings
- Use [numpy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) for all public functions, classes, and modules
- Check that your docstrings are formatted correctly with [pydocstyle](https://www.pydocstyle.org/en/stable/):
  ```bash
  pydocstyle --convention=numpy puncturedfem tests
  ```

### MkDocs
- If adding a new function/class/module, include a markdown file in the appropriate subdirectory of `doc/src`
- Add the new file to the `nav` section of `mkdocs.yml`
- If you are contributing to the documentation, you can use [MkDocs](https://www.mkdocs.org/) to preview your changes:
  ```bash
  mkdocs build
  mkdocs serve
  ```
- Open a web browser and navigate to `http://127.0.0.1:8000/` to preview the documentation

## Commit your changes
- Stage your changes
  ```bash
  git add file1 file2 ...
  ```
  - Don't add any files you haven't directly edited (e.g. don't add your virtual environment files)
  - Do you want `git` to [ignore certain files or directories](https://git-scm.com/docs/gitignore)?
  - Did you [accidentally add a file](https://git-scm.com/docs/git-reset)?
- Commit your changes
  ```bash
  git commit -m "a short note about the changes"
  ```
  - Do you need to [undo a commit](https://www.atlassian.com/git/tutorials/undoing-changes)?

## Push your changes
- Once you are happy with the state of your code and are ready to merge them with the `main` branch, push your changes to your fork:
  ```bash
  git push
  ```
- Finally, [open a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) on GitHub to merge your changes into the `main` branch of the parent repository

## Release (maintainers only)
### Create a release branch
- Ensure that `main` is up-to-date:
  ```bash
  git switch main
  git pull
  ```
- Create a new branch for the release:
  ```bash
  git switch -c release-vX.Y.Z
  ```
### Update the version number and changelog
- Update the version number in `pyproject.toml`
- Update the `doc/ROADMAP.md`
  - Move completed items to the `CHANGELOG.md`
- Update the `CHANGELOG.md`
  - Version number
  - Date
  - List of changes
- Commit the changes:
  ```bash
  git commit -m "Release vX.Y.Z"
  ```
- Push the changes:
  ```bash
  git push --set-upstream origin release-vX.Y.Z
  ```
- Open a pull request to merge the release branch into `main`
### Merge the release branch
- Once the pull request is merged, update the `main` branch:
  ```bash
  git switch main
  git pull
  ```
- Tag the release:
  ```bash
  git tag -a vX.Y.Z -m "Release vX.Y.Z"
  ```
- Push the tag:
  ```bash
  git push --tags
  ```
- Publish the release on GitHub

### Publish to PyPI
- Build the distribution:
  ```bash
  poetry build
  ```
- Publish the distribution:
  ```bash
  poetry publish
  ```
