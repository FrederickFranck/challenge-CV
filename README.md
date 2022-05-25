# Mole Detection App

# Description

This a web app that classifies moles based on images.

# Usage

Website: https://mole-doctor.herokuapp.com/

# Installation


Install required packages

```bash
pip install -r project/requirements.txt
```

To build and save the model, this needs to be done at least once before hosting the app locally or in docker
```bash
python model_building.py
```

Host the app locally, it will be running on [localhost](http://localhost:5000/)

```bash
python app.py
```

## Docker

Build the docker image
```bash
docker build . -t mole-doctor
```

Deploy the docker image to a container and run locally.
App will be running on [localhost](http://localhost:5000/)

```bash
docker run -d -p 5000:5000 mole-doctor
```

# Work in progress...