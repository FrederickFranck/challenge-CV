FROM python:3.8-slim-buster
COPY ./project/docker_requirements.txt /
RUN pip install --no-cache-dir -r docker_requirements.txt
COPY ./project /project
WORKDIR /project
CMD ["python", "app.py"]