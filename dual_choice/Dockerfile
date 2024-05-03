# copypase for now

FROM python:3.12

WORKDIR /app

COPY requirements.lock .

RUN sed -i '/^-e//d' requirements.lock

RUN pip install --no-cache-dir -r requirements.lock

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
