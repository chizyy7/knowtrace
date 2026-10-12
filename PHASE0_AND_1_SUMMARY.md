# KnowTrace Project - Phase 0 & 1 Summary (Windows-Focused)

## Phase 0: Scaffolding - COMPLETED

### Files Created/Updated:
- `.gitignore`: Standard Python/data ignores + `.gitkeep` exceptions
- `README.md`: Project overview with Windows installation instructions
- `LICENSE`: MIT license
- `pyproject.toml`: Setuptools configuration with `where = ["src"]`
- `requirements.txt`: Core dependencies (numpy, pandas, scikit-learn, torch, etc.)
- `Makefile`: Setup, test, and clean targets
- `src/knowtrace/__init__.py`: Package version (`__version__ = "0.1.0"`)
- `src/knowtrace/utils/__init__.py`: Utils package init
- `src/knowtrace/utils/config.py`: YAML configuration loader
- `src/knowtrace/utils/seed.py`: Reproducibility seeding function
- `src/knowtrace/utils/logger.py`: Stream logger with standard format
- `src/knowtrace/data/__init__.py`: Data package init
- `src/knowtrace/models/__init__.py`: Models package init
- `src/knowtrace/train/__init__.py`: Train package init
- `src/knowtrace/eval/__init__.py`: Eval package init
- `src/knowtrace/app/__init__.py`: App package init
- `tests/__init__.py`: Tests package init
- `tests/test_setup.py`: Placeholder test to verify test framework
- Directory structure: All required folders created

### How to Run Phase 0:
```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install package in development mode
pip install -e .

# Run tests
pytest tests/ -v
```

### Expected Output:
- Successful virtual environment creation and activation
- Package installs without errors
- 1 test passes (test_setup.py)

### Design Choices:
- Used `pyproject.toml` with setuptools for proper package installation
- Virtual environment activation follows Windows standard (`.\.venv\Scripts\Activate.ps1`)
- Logger provides consistent formatting with timestamps and levels
- Seed function ensures reproducibility across random, numpy, and torch
- All `__init__.py` files make packages importable

## Phase 1: Data Pipeline - COMPLETED

### Files Created:
- `scripts/download_data.ps1`: PowerShell script for ASSISTments download
  - Tries Kaggle CLI first (`kaggle datasets download -d nicolasw/assistments-2009-2010`)
  - Falls back to manual download instructions
  - Provides synthetic fallback option
- `src/knowtrace/data/load.py`: 
  - `COLUMN_ALIASES` mapping for robust column name handling
  - `load_raw()` function to load and standardize CSV data
  - `generate_synthetic()` function for fallback data generation
  - CLI interface: `python -m knowtrace.data.load --synthetic`
- `src/knowtrace/data/preprocess.py`:
  - `PreprocessConfig` dataclass for configurable preprocessing
  - User-based train/val/test split (prevents leakage)
  - ID mapping with index 0 reserved for padding
  - Filtering for low-activity users and rare skills
  - CLI interface: `python -m knowtrace.data.preprocess --synthetic`
- `src/knowtrace/data/dataset.py`:
  - `KTDataset` class returning padded tensors
  - `pad_collate` function for DataLoader
  - `load_sequences()` helper to read JSONL files
  - Returns dictionary: `q_ids`, `skill_ids`, `corrects`, `mask`
- `notebooks/01_eda.ipynb`: Exploratory data analysis notebook
  - Sequence length distribution analysis
  - Interactions-per-user histogram
  - Attempts-per-problem and attempts-per-skill analysis
  - Per-skill accuracy distribution
  - Train/val/test user overlap verification

### How to Run Phase 1:
```powershell
# 1. Generate synthetic data (or download real data via download_data.ps1)
python -m knowtrace.data.load --synthetic

# 2. Preprocess the data
python -m knowtrace.data.preprocess --synthetic

# 3. Verify the processed data was created
dir data\processed
# Should see: train.jsonl, val.jsonl, test.jsonl, id_maps.json

# 4. Run the EDA notebook
jupyter notebook notebooks\01_eda.ipynb
```

### Expected Output:
- Synthetic data generation: ~200 users, 15 skills, 60 problems, 80 max seq length
- Processed data: Train/val/test split by user (default 80/10/10)
- JSONL files with one user sequence per line
- ID maps showing mapping from original IDs to internal indices (starting at 1)
- EDA notebook showing distributions and verifying no user leakage

### Design Choices:
- **PowerShell scripts**: Used native Windows shell instead of bash
- **Synthetic fallback**: Ensures pipeline runs without external dependencies
- **User-based splitting**: Prevents data leakage by keeping all interactions of a user in one split
- **Index 0 reserved for padding**: Standard practice in NLP/sequential models
- **Configurable preprocessing**: All parameters adjustable via CLI or config
- **Comprehensive EDA**: Notebook validates data quality and split integrity

## Next Steps:
Proceed to Phase 2: BKT Baseline implementation