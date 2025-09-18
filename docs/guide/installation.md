
# Installation and Setup

## Prerequisites
- **Python**: Version 3.11 or later.
- **IDE**: [Visual Studio Code](https://code.visualstudio.com/) is recommended.
- **Git**: Either [Git](https://git-scm.com/) (command line) or [GitHub Desktop](https://desktop.github.com/).

## Creating a Virtual Environment in VS Code
1. Open the C2F4DT folder in VS Code.
2. Open the terminal (`Ctrl+``).
3. Create a virtual environment:
  ```bash
  python -m venv venv_c2f4dt
  ```
4. Activate the environment:
  - On macOS/Linux:
    ```bash
    source venv_c2f4dt/bin/activate
    ```
  - On Windows:
    ```bash
    venv_c2f4dt\Scripts\activate
    ```

VS Code should auto-detect the `venv_c2f4dt`. If not, select it manually (`Python: Select Interpreter`).

## Installing Packages
Once the environment is active, install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```


## Activating in Regular or Development Mode

### Regular Mode
To activate the application in regular mode, run:
```bash
pip install .
```

### Development Mode
For development mode, which includes additional tools and dependencies for development, run:
```bash
pip install -e .[dev]
```

You can also activate in user mode (without requiring administrative privileges):
```bash
pip install --user .
```


## Running the Application
- To launch the main app:
  ```bash
  python main.py
  ```
- Or run example scripts:
  ```bash
  python examples/run_app.py
  ```

---

# Version Control with Git

## Cloning the Repository
```bash
git clone https://github.com/gcastellazzi/C2F4DT.git
cd C2F4DT
```

If using GitHub Desktop, choose “Clone repository” and paste the repo URL.

## Creating a Branch
Always create a feature branch before editing:
```bash
git checkout -b feature/my-change
```

## Making Commits
- Keep commits small and focused.
- Use descriptive messages:
  ```bash
  git commit -m "fix: corrected normals orientation in slice viewer"
  ```
- **Do not commit**:
  - `venv_c2f4dt/`
  - large datasets
  - build artifacts

Make sure `.gitignore` includes `venv_c2f4dt/`, `__pycache__/`, and `site/`.

## Pushing Changes
```bash
git push origin feature/my-change
```

Then open a Pull Request on GitHub.

