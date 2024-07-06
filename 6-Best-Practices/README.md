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
In this section the code that was used for predicting on AWS using Kinesis and Lambda from Section 4 of the class, will be tested. The original code can be found here: [streaming](https://github.com/joweyel/mlops-zoomcamp/tree/main/04-deployment/streaming). The modified version of the code can be found in the directory [code](./code/).
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
  
Now open the pipenv and run pytest.
```bash
pipenv shell
pytest
```

The next step is to open the Test-Sidebar in VS-Code (the retort symbol)
- Click on `Configure Python Tests` an select the `tests` directory

Now it's time to create tests:




<a id="2-integration-test"></a>
## 6.2 Integration tests with docker-compose

<a id="3-local-stack"></a>
## 6.3 Testing cloud services with LocalStack

<a id="4-linting"></a>
## 6.4 Code quality: linting and formatting

<a id="5-git"></a>
## 6.5 Git pre-commit hooks

<a id="6-make"></a>
## 6.6 Makefiles and make

<a id="7-homework"></a>
## 6.7 Homework
