# KnowTrace — Adaptive Knowledge Tracing with Deep Learning

>[!NOTE]
> This project is a work in progress. See the [roadmap](#roadmap) for planned features.

## Overview

KnowTrace implements knowledge tracing models to predict the probability of a student answering the next question correctly given their sequence of past attempts. It also predicts question difficulty from text (or historical accuracy) and uses it as an auxiliary feature.

## Roadmap

- [x] Phase 0: Scaffolding
- [ ] Phase 1: Data Pipeline
- [ ] Phase 2: BKT Baseline
- [ ] Phase 3: Deep Knowledge Tracing (DKT)
- [ ] Phase 4: DKT+ (Auxiliary Features)
- [ ] Phase 5: Question Difficulty Prediction
- [ ] Phase 6: Evaluation & Reporting
- [ ] Phase 7: Streamlit Demo
- [ ] Phase 8: Packaging & Deployment

## Installation

```powershell
# Clone the repository
git clone https://github.com/chizyy7/knowtrace.git
cd knowtrace

# Create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -e .
```

## Usage

See each phase's instructions for running specific components.

## License

MIT