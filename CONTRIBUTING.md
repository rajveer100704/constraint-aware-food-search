# Contributing to Constraint-Aware Hybrid Food Search Engine

Thank you for considering contributing to the Constraint-Aware Hybrid Food Search Engine!

## Development & Testing Workflow

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rajveer100704/constraint-aware-food-search.git
   cd constraint-aware-food-search
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the master build pipeline**:
   ```bash
   python build.py
   ```

4. **Run unit & integration tests**:
   ```bash
   pytest -v
   ```

## Pull Request Guidelines

- Ensure all 22 pytest unit tests pass before submitting a pull request.
- Keep commits structured and clean.
- Update documentation and ADRs if modifying core retrieval or ranking logic.
