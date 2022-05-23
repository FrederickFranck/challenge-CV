FROM alpine:latest
RUN apt-get update -y && apt-get install -y --no-install-recommends\
        python3-pip \
        python-is-python3 \
        build-essential \
   && rm -rf /var/lib/apt/lists/*
COPY ./project /project
WORKDIR /project
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]