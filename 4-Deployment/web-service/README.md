## Deploying a model as a web-service

### Creating a virtual environment with Pipenv

<!-- 
```bash
pipenv install scikit-learn==1.2.2 flask --python=3.9
# Opening the environment with
pipenv shell
``` 
-->

```bash
pipenv install scikit-learn==1.2.2 flask --python=3.9
# Opening the environment with
pipenv shell
```

- The generated Pipfile contains the following

```log
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
scikit-learn = "==1.2.2"
flask = "*"
requests = "*"
gunicorn = "*"
numpy = "==1.26.4"

[dev-packages]
requests = "*"

[requires]
python_version = "3.9"
python_full_version = "3.9.19"
```
To avoid `scikit-learn` and `numpy` errors concerning `numpy.dtype` the last version before `numpy 2.0.0` is used.

### Creating a script for prediction  
```bash
touch predict.py test.py
```

### Putting the script into a Flask app

- See [predict.py](predict.py) and [test.py](test.py) for exemplary scripts for querying a model for prediction
- Using a production Server instead of development Server (locally)
    - ```bash
      pipenv install gunicorn
      ```
    - Running the `gunicorn` server
      ```bash
      # run the app from predict file
      gunicorn --bind=0.0.0.0:9696 predict:app
      python3 test.py
      ```
- There could arise errors because libraries are not available in base-python environment and the web-app environment
    - `requests`-packag is only required for development
    ```bash
    pipenv install --dev requests
    ```

### Packaging the app to Docker

- Building the `Dockerfile` as a self-contained and portable environment. 
```dockerfile
FROM python:3.9.19-slim  # use a specific python version

# Updata pip and install pipenv
RUN pip install -U pip 
RUN pip install pipenv

# Specifying workdir
WORKDIR /app

# Copy Pipfile files to the container
COPY [ "Pipfile", "Pipfile.lock", "./"]

# Install dependencies (globally) in docker container
RUN pipenv install --system --deploy

# Copy the relevant ml scripts to the container
COPY ["predict.py", "lin_reg.bin", "./"]

# Make 9696 available for port forwarding
EXPOSE 9696

# Command to be run, when starting the container
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]

```

- Building the Docker-Container
```bash
docker build -t ride-duration-prediction-service:v1 .
```

```bash
# -it: Interactive Mode
# --rm: Removes the volume after running
# -p: Port forwarding
docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v1
```

- Testing the dockerized model with test-example:
```bash
python3 test.py
```