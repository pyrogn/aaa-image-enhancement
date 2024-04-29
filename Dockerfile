# not tested
FROM python:3.10-slim

WORKDIR /app

# if we copy, we should remove one line like <(requirements.lock | sed '/^-e/d')
# COPY requirements.lock .
# RUN pip install --no-cache-dir -r requirements.lock

COPY . .

RUN pip install .

CMD ["uvicorn", "src.aaa_image_enhancement.main:app", "--host", "0.0.0.0", "--port", "8000"]