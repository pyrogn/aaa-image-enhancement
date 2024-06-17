FROM python:3.12-slim

WORKDIR /app
COPY ./requirements.lock README.md ./pyproject.toml /app/

RUN pip install uv

COPY ./src /app/src
COPY ./tests /app/tests

RUN uv pip install --system -e .[test]
RUN uv pip install --system opencv-contrib-python-headless

CMD ["uvicorn", "src.aaa_image_enhancement.apps.main:app", "--host", "0.0.0.0", "--port", "8000"]
