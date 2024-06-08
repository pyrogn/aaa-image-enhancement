# not tested
FROM python:3.12-slim

WORKDIR /app
COPY ./requirements.lock README.md ./pyproject.toml /app/

RUN pip install uv

COPY ./src /app/src
RUN uv pip install --system -e .

CMD ["uvicorn", "src.aaa_image_enhancement.app:app", "--host", "0.0.0.0", "--port", "8000"]
