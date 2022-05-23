FROM python:3.8-slim
COPY ./project /project
WORKDIR /project
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "app.py"]