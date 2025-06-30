## Q1. Refactoring

The new file can be found here: [batch2.py](homework/batch2.py)

## Q2. Installing pytest

```bash
# Go to homework folder (if not already there)
cd homework

# Install pytest
pipenv install --dev pytest

# Create tests-folder and go to it
mkdir -p tests && cd tests

# Create test_batch.py
touch test_batch.py

# Copy model-data
cp ../model.bin .
```