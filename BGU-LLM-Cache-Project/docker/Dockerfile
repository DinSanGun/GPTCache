FROM python:3.11-slim

WORKDIR /app/GPTCache/BGU-LLM-Cache-Project
COPY . /app/GPTCache

RUN pip install -U pip && pip install -r BGU-LLM-Cache-Project/requirements.txt

CMD ["python","-m","src.bench.runner","--config","configs/mock.yaml"]
