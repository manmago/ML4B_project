# ML4B Sleep Classification Project

## Setup & Getting Started

### Prerequisites
- Python 3.11+
- UV package manager
- Git

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/manmago/ML4B_project.git
cd ML4B_project
```

#### 2. Set Up Virtual Environment

**Windows (VS Code Terminal - PowerShell):**

VS Code's terminal defaults to PowerShell. Run:

```powershell
# One-time setup (first time only):
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Create and activate environment:
uv sync
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

If you prefer Command Prompt, use `.bat` instead:

```cmd
uv sync
.venv\Scripts\activate.bat
```

**Mac/Linux:**
```bash
uv sync
source .venv/bin/activate
```

#### 3. Verify Setup

Once activated, you should see `(.ml4b)` in your terminal. Test with:
```bash
python --version
uv --version
```

## Demo Nights

The Streamlit app reads curated demo nights from `data/example_nights/`.

- Committed demo data should be compact `.joblib` bundles such as `data/example_nights/DEMO.joblib`.
- Raw CSV demo folders are for local regeneration only and are ignored by Git.
- If you rebuild the bundle from raw sensor exports, keep the raw source outside the repository or leave it ignored.
