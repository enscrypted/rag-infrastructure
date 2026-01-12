FROM python:3.11-slim

WORKDIR /app

# Install torch CPU first (separate step with correct syntax)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# Install other dependencies
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pymongo \
    motor \
    numpy \
    pydantic \
    python-multipart \
    jinja2 \
    aiofiles \
    sentence-transformers

# Copy application files
COPY app.py /app/
COPY templates/ /app/templates/
COPY static/ /app/static/

EXPOSE 8090

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8090", "--reload"]