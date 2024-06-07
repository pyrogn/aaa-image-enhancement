FROM python:3.12-slim

WORKDIR /app

COPY ./requirements.lock README.md ./pyproject.toml ./demo /app/
COPY ./src /app/src

RUN pip install uv
RUN uv pip install --system -e .
RUN uv pip install --system opencv-contrib-python-headless

EXPOSE 5000

CMD ["python", "-m", "demo_app"]
