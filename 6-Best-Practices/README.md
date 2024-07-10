# 6. Best Practices

- 6.1 [Testing Python code with pytest](#1-pytest)
- 6.2 [Integration tests with docker-compose](#2-integration-test)
- 6.3 [Testing cloud services with LocalStack](#3-local-stack)
- 6.4 [Code quality: linting and formatting](#4-linting)
- 6.5 [Git pre-commit hooks](#5-git)
- 6.6 [Makefiles and make](#6-make)
- 6.7 [Homework](#7-homework)

<a id="1-pytest"></a>
## 6.1 Testing Python code with pytest
In this section the code that was used for predicting on AWS using Kinesis and Lambda from Section 4 of the class, will be tested. The original code can be found here: [streaming](https://github.com/joweyel/mlops-zoomcamp/tree/main/04-deployment/streaming). 

![section4_streaming](imgs/6_1_1_streaming_architecture.jpg)

### Setup

The modified version of the code can be found in the directory [code](./code/).
The goal of this section is to test the code with unit tests.

Everything here is done in the [code](./code/) directory.
 
Installing the *Pipenv* and test-dependencies:
```bash
pipenv install
pipenv install --dev pytest
```
For testing with VS-Code: 
- Install the `Python`-extension
- Select python interpreter with `[Ctrl]+[Shift]+[P]`, then search for `Python: Select Interpreter`
  - *Manually configure the interpreter*
    - Find out path to pipenv: `pipenv --venv`
    - Insert path into `+ Enter interpreter path` and append `/bin/python`
  - *Find Interpreter in Interpreter list (is often already there)*
    - Just click on it!
  - Create `tests` directory in the `code` directory
  
Now open the pipenv and run pytest.
```bash
pipenv shell
pytest
```

### Configure PyTest

The next step is to open the Test-Sidebar in VS-Code (the retort symbol)
- Click on `Configure Python Tests` an select the `tests` directory

Now it's time to test the cpde. The tests can be found in the files of the [`tests`](code/tests/)-directory. While the content from section 4's Streaming folder was initially used it was subsequently adapted to be more suitable to be tested.

The relevant files of this sction are:
- [lambda_function.py](code/lambda_function.py)
- [model.py](code/model.py)
- [model_test.py](code/tests/model_test.py)
- [Dockerfile](code/Dockerfile)
- [test_docker.py](code/test_docker.py)
  

<a id="2-integration-test"></a>
## 6.2 Integration tests with docker-compose

**Types of testing:**
- `Unit-Tests`: Tests a single small "unit" in the code usually functions, methods, etc. by themselves
- `Integration-Tests`: Tests the integration of a code segment into the overall code s.t. the interaction of code with other components works as expected



<a id="3-local-stack"></a>
## 6.3 Testing cloud services with LocalStack

This section handles the testing of cloud ressurces that were left out in the unit- and integration-test sections. For this [localstack](https://github.com/localstack/localstack) is used.

To incorporate `localstack` into the code, `docker-compose` is used. Therefore the previously used docker-compose configuration is extended by the service `kinesis`:

```yaml
services:
  backend:
    image: ${LOCAL_IMAGE_NAME}
    ports:
      - "8080:8080"
    environment:
      - PREDICTIONS_STREAM_NAME=ride_predictions
      - TEST_RUN=True
      - RUN_ID=Test123
      - AWS_DEFAULT_REGION=us-east-1
      - AWS_PROFILE=mlflow-user
      - MODEL_LOCATION=/app/model
    volumes:
      - ~/.aws:/root/.aws
      - ./model:/app/model
  kinesis:
    image: localstack/localstack
    ports: 
      - 4566:4566
    environment:
      - SERVICES=kinesis
```
To pull the `localstack`-image (at first start) and to run the `kinesis`-service the following command has to be used:
```bash
docker-compose up kinesis
```

The goal now is to use local versions of AWS services, that are simulated by `localstack`. To see what `kinesis`-streams are currently running you can use this command:
```bash
aws kinesis list-streams
```
If there are any, you can delete them since they are not used here and are costing you money even in idle-state. If all streams are deleted you should get this result from `aws kinesis list-streams`:
```json
{
    "StreamNames": [],
    "StreamSummaries": []
}
```
Now it's time to connect to the kinesis-stream that was started with the localstack-container.
```bash
aws --endpoint-url=http://localhost:4566 kinesis list-streams
```
If everything worked out, you should get this output:
```json
{
    "StreamNames": []
}
```

Now create a stream with localstack:
```bash
aws --endpoint-url=http://localhost:4566 \
    kinesis create-stream \
    --stream-name ride_predictions \
    --shard-count 1
```

You should now see the strem in the stream list in localstack:
```bash
aws --endpoint-url=http://localhost:4566 kinesis list-streams
{
    "StreamNames": [
        "ride_predictions"
    ]
}
```


To obtain predictions from the stream the following commands can be used:

```bash
export SHARD='shardId-000000000000'
export PREDICTIONS_STREAM_NAME='ride_predictions'

# Get shard iterator
SHARD_ITERATOR=$(aws --endpoint-url=http://localhost:4566 \
    kinesis get-shard-iterator \
    --shard-id ${SHARD} \
    --shard-iterator-type TRIM_HORIZON \
    --stream-name ${PREDICTIONS_STREAM_NAME} \
    --query 'ShardIterator' \
)

# Extracting records from shard
RESULT=$(aws --endpoint-url=http://localhost:4566 kinesis get-records --shard-iterator $SHARD_ITERATOR)

# Getting the predictions and decode them
echo $RESULT | jq -r '.Records[0].Data' | base64 --decode
```

The result of a test run returned the following output:
```json
{
    "model": "ride_duration_prediction_model", 
    "version": "Test123", 
    "prediction": {
      "ride_duration": 21.432393319299262, 
      "ride_id": 256
    }
}
```

To make everything more compact and testable instead of running every command in the commmand line, everything will be packaged in the python script [test_kinesis.py](code/integration-test/test_kinesis.py). The test is also included in [run.sh](code/integration-test/run.sh):

```bash
...
pipenv run python3 test_kinesis.py 

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi

docker-compose down
```

For re-execution of `run.sh`, docker-compose has to be stopped first.
```bash
docker-compose down
./run.sh
```

### Conclusion
To run the created tests you can now use the following commands (in `code`-directory):
```bash
# Unit tests
pipenv run pytest tests/
# Integration tests
./integration-test/run.sh
```


<a id="4-linting"></a>
## 6.4 Code quality: linting and formatting

<u>The two important concepts of this section:</u>
- **Linting:** Process of analyzing source code to identify potential errors, improve code quality, and enforce coding standards
- **Formatting:** Deals with the visual arrangement of code, including aspects like indentation, line breaks, and spacing

### Installing `Pylint` to lint Python code
```bash
pipenv install --dev pylint


pipenv shell      # Activate environment
pylint model.py   # Example for linting

# Linting all the code recursively starting from "."
pylint --recursive=y .
```

It is also possible to select and configure a linter in VSCode by installing the VSCode extension of the linter `Pylint`. 

### Excluding certain warnings from the linter (3 different ways)
- **Version 1 (VSCode config):**
  - To exclude certain warnings, go to the settings by typing `[Ctrl]+[,]` and search for pylint. In the **`args`**-part of the pylint settings you can disable warnings by typing `--disable=<Warning-ID>`. 
- **Version 2 (.pylintrc)**:
  - create the file [`.pylintrc`](code/.pylintrc)  in the `code` directory and add the following code to disable certain warnings
    ```ini
    [MESSAGE CONTROL]
 
    disable=missing-function-docstring,
            missing-final-newline,
            missing-class-docstring
    ```
- **Version 3 (pyproject.toml)**:
  - Allows to configure python projects, including linting:
    ```ini
    [tool.pylint.message_control]
    disable = [ # warnings to exclude
        "missing-function-docstring",
        "missing-final-newline",
        "missing-class-docstring",
        "missing-module-docstring"
    ]
    ```

To explicitly exclude pylint warnings in specific functions / methods / classes locally, you can just add the following to the code of the component:
```python
# Example of locally disabling warnings
def get_model_location(run_id):
    # pylint: disable=missing-function-docstring
    ...
    return model_location
```

### Code formatting with `black` and `isort`

- **`black`**: A code formatter for python code
- **`isort`**: Library for sorting python imports according to some guideline + compatibility with `black`

Parameterization of `black` is done via parameter or via `pyproject.toml`
```bash
black --line-length 80 --skip-string-normalization --target-version py39 
```

```ini
[tool.black]
line-length = 80
target-version = ['py39']
skip-string-normalization = true
```

Now `black` can be run with the following command:
```bash
black .  # already parameterized by `pyproject.toml` in [tool.black]
```

To apply `isort` the same syntax as black can be used:
```bash
isort . # already parameterized by `pyproject.toml` in [tool.isort]
```

### Conclusion
Linting, formatting and sorting of imports can all be done automatically, given some rules. This makes them very easy to use in CI/CD pipelines, in which they are often used.
- **Linting - `pylint`**: Assures code quality
- **Formatting - `black`**: Assumes formatting according to a given set of rules
- **Sorting - `isort`**: Sorts imports in the code according to a given set of rules
  

When putting everything together (after configuring each component), you get commands like this:
```bash
isort .
black .
pylint --recursive=y .
pytest tests
```

<a id="5-git"></a>
## 6.5 Git pre-commit hooks
When committing code to Git it should habe ideally been processed with the 4 commands above. However manually calling them every time it is possible to automate `isort`, `black` and `pylint`. For this <u>Git uses pre-commit hooks</u>. 

The processing via pre-commit hooks can be done the python package `pre-commit`:
```bash
pipenv install --dev pre-commit
```

To use it in a specific (sub-)folder of the Git-Repo you are currently working in you can initialize an "empty" repo in this folder with `git init`. Normally hooks are handled in the repos configuration, however the need to only apply the hooks only to one folder makes this not viable. This local repository is however only temporary and `.git` is removed after this section is done. 

```bash
cd code
git init
```

Running `pre-commit` for first time and getting a sample pre-commit config:
```bash
pre-commit sample-config  # writes to stdout
pre-commit sample-config > .pre-commit-config.yaml  # write to file
```

Next step is to create a pre-commit hook in the git-folder `.git/hooks/`:
```bash
pre-commit install
# Looking into the file 
less .git/hooks/pre-commit
```

The step of installing pre-commit is required on every computer that wants to use them since the `.git` folder is local only and is updated with newer commits from the repo.

**Create `.gitignore`, `add` and `commit files`**

1. Create gitignore file that excludes `__pycache__`
2. Run `git add .` to add all files from the current folder
3. Run `git commit -m "initial commit"` to trigger the pre-commit hooks
4. Look at `git diff` to see the differences
5. Run `git add .` to add the changed files to be committed
6. Run `git commit -m "fixes from pre-commit default hooks"` to commit the changed files

**Adding `isort` and `black`, `pylint` and `pytest` to the hooks**

```yaml
# See https://pre-commit.com for more information
# See https://pre-commit.com/hooks.html for more hooks
repos:
 - repo: https://github.com/pre-commit/pre-commit-hooks
   rev: v4.6.0
   hooks:
    - id: trailing-whitespace
    - id: end-of-file-fixer
    - id: check-yaml
    - id: check-added-large-files
 - repo: https://github.com/pycqa/isort
   rev: 5.13.2
   hooks:
    - id: isort
      name: isort (python)
 - repo: https://github.com/psf/black
   rev: 24.4.2
   hooks:
     - id: black
       language_version: python3.9
 - repo: local
   hooks:
     - id: pylint
       name: pylint
       entry: pylint
       language: system
       types: [python]
       require_serial: true
       args:
         [
           "-rn", # Only display messages
           "-sn", # Don't display the score
           "--recursive=y"
         ]
 - repo: local
   hooks:
    - id: pytest-check
      name: pytest-check
      entry: pytest
      language: system
      pass_filenames: false
      always_run: true
      args: [
        "tests/"
      ]
```

If you are getting errors with [`pyproject.toml`](code/pyroject.toml) there is a possibility that you specified the wrong versions of the used libraries in the hooks. To remedy this you can automatically update / change to the library version you have with `pre-commit autoupdate`.

When a file files pytest then it will not be moved to the commit stage but still remains in the `untracked` / `modified` state.

Now you can delete the `.git` folder to redo it again an test if the commit goes through:
```bash
rm -rf .git
pre-commit intall
git add .
git commit -m "initial commit"
```

<a id="6-make"></a>
## 6.6 Makefiles and make

<a id="7-homework"></a>
## 6.7 Homework
