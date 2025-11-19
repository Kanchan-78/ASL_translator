
# ASL Sign Language Recognition

TranslateMe project implements an **American Sign Language (ASL) recognition system** using a **RandomForestClassifier** trained on a custom dataset. The system includes machine learning model development, data preprocessing, a FastAPI backend, and supporting components for real-time prediction.

-----

### Instructions for installation of UV

> 1.  **Install uv**:
>       * **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
> 2.  **Open a terminal**
> 3.  **Verify Installation**
>     ```bash
>     uv --version
>     ```

-----

### Install C++ Build Tools

1.  **Download:** Go to the [Visual Studio Downloads page](https://visualstudio.microsoft.com/downloads/).

2.  **Find Tools:** Scroll down to "Tools for Visual Studio" (or "All Downloads" -\> "Tools for Visual Studio").

3.  **Install:** Download and run the "Build Tools for Visual Studio".

4.  **Select Workload:** When the installer opens, you **must** select the **"Desktop development with C++"** workload. This includes the C/C++ compilers and Windows SDKs needed.

5.  **Restart:** After the installation is complete, **restart your computer** (or at least restart your terminal/VSCode) to ensure the new environment variables (like the `PATH` to the compiler) are loaded.

6.  **Re-run Install:** Go back to your project directory and try the command:

    If got fresh new project
    ```bash
    uv pip install -r requirements.txt
    ```
    or If got shared project
    ```bash
    uv sync
    ```

-----

The beauty of `uv` is that it makes sharing projects reproducible.

-----

### How to share uv project 

**Do this on your PC first:**

1.  **Import requirements into pyproject.toml:**
    Run this command in the terminal to read `requirements.txt` and officially add them to `uv` project:

    ```bash
    uv add -r requirements.txt
    ```

2.  **Verify the cleanup:**
    Check the `pyproject.toml`. It should now look populated, like this:

    ```toml
    [project]
    name = "translateme"
    dependencies = [
        # ... all packages ...
    ]
    ```

3.  **Update the Lockfile:**
    Run this to ensure `uv.lock` captures exactly what is installed:

    ```bash
    uv sync
    ```

-----

## 🤝 Contributions

Pull requests, improvements, and suggestions are welcome.
