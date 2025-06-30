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

# Create test_batch.py and "__init__.py"
touch test_batch.py __init__.py
```

For importing `batch2.py` the file `__init__.py` is needed!


## Q3. Writing first unit test

**Test-Code**
- [`test_batch.py`](homework/tests/test_batch.py)
- `Number of expected rows`: 2


## Q4. Mocking S3 with Localstack

