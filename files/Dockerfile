FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python data/generate_synthetic_data.py
RUN python src/tag_and_probe.py
RUN python src/uncertainties.py
RUN python src/scale_factors.py

CMD ["python", "src/scale_factors.py"]
