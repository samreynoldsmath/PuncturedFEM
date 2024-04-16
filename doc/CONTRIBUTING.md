# Contributing Guide
This is a brief guide describing how to contribute to the `PuncturedFEM` project.

### TL;DR
- Fork this repo
- Checkout a new branch
- Create and activate a virtual environment
- Install dev tools
- Install this package in edit mode
- Use `pylint` and `mypy` for linting
- Run tests with `pytest`
- Use `isort` and `black` (in that order) for formatting
- Clear all Jupyter notebook outputs
- Commit, push to your fork, and make a pull request

## Getting started

### Prerequisites
- [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- [python3](https://www.python.org/)
- [pip](https://pypi.org/project/pip/)
- [venv](https://docs.python.org/3/library/venv.html)

### Fork & clone
- [Fork this repo](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) and make sure it is up-to-date
  - **Note:** If you forget to make a fork and clone this repo directly, [you can fix it](#did-you-fork)
- Clone your fork to your local machine:
  ```bash
  git clone https://github.com/<myusername>/PuncturedFEM <my/path/to/PuncturedFEM>
  ```

### Set up virtual environment
- Configure a virtual environment:
  ```bash
  python3 -m venv <my/path/to/PuncturedFEM>
  ```
- Change directory to your local repo:
  ```bash
  cd <my/path/to/PuncturedFEM>
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
### Install packages
- Use `pip` to install developer tools
  ```bash
  pip install -r requirements-dev.txt
  ```
- Install this package in edit mode
  ```bash
  pip install -e .
  ```

## Make your changes
- Make sure that `main` is up-to-date:
  ```
  git checkout main
  git pull
  ```
- Checkout a new branch, named according to the contribution you wish to make (short but descriptive is best)
  ```bash
  git checkout -b <descriptive-branch-name>
  ```
- Make your changes and save your files
- It is best practice to make small, incremental changes and make [commits](#commit-your-changes) as you go
  - This way if you make a mistake, it is easy to [return to a version of your code that works](https://www.atlassian.com/git/tutorials/undoing-changes)

## Linting, testing, & formatting
### Run linters
If you have edited the source code or tests, use `pylint` to find errors and use `mypy` to find type errors.
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


### Run tests
- Be sure that your virtual environment is active
- Use [pytest](https://docs.pytest.org/en/8.0.x/) to run the tests in the `test/` directory
  ```bash
  pytest tests
  ```
- Optionally, you can use the `-s` flag to see print statements (helpful for debugging)
  ```bash
  pytest -s tests
  ```
- Be sure all tests pass before proceeding

### Run formatters
This step helps make the code you've written more readable and standardized.
- Be sure that your virtual environment is active
- If contributing Jupyter notebooks, clear all outputs:
  ```bash
  jupyter nbconvert --clear-output --inplace examples/*.ipynb
  ```
- If you have edited the source code, format import statements with [isort](https://pypi.org/project/isort/):
  ```bash
  isort puncturedfem
  ```
  - **WARNING:** Running `isort` on the `tests/` and `examples/` directories can break them.
  - Always run `black` after running `isort`
- Format source code and tests with [black](https://pypi.org/project/black/):
  ```bash
  black puncturedfem tests
  ```
- We highly recommend that you run `black` on any Jupyter notebook examples you have written

## Commit your changes
- Stage your changes
  ```bash
  git add <file1 file2 ...>
  ```
  - Don't add any files you haven't directly edited (e.g. don't add your virtual environment files)
  - Do you want `git` to [ignore certain files or directories](https://git-scm.com/docs/gitignore)?
  - Did you [accidentally add a file](https://git-scm.com/docs/git-reset)?
- Commit your changes
  ```bash
  git commit -m "a short note about the changes"
  ```
  - Do you need to [undo a commit](https://www.atlassian.com/git/tutorials/undoing-changes)?

### Did you fork?
If you forgot to fork this repository and cloned it directly, you will need to:
- [Make a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)
- Link your local repo to your fork:
  ```bash
  git remote set-url origin https://github.com/<myusername>/PuncturedFEM.git
  ```

### Push your changes
- Once you are happy with the state of your code and are ready to merge them with the `main` branch, push your changes to your fork:
  ```bash
  git push
  ```
  - Did you [configure git](https://git-scm.com/book/en/v2/Customizing-Git-Git-Configuration)?
- Finally, [create a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) to merge your changes into the `main` branch

## Resources
- [Git Graph](https://marketplace.visualstudio.com/items?itemName=mhutchie.git-graph): a VS Code plugin