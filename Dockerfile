  FROM python:3.10.11-bullseye
  COPY . .
  RUN pip install --upgrade pip
  RUN pip install -r requirements.txt
  RUN pip install .
  CMD uvicorn pagodas.api.fast:app --reload --host 0.0.0.0 --port $PORT
