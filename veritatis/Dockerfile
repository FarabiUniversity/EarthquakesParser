# Use Python 3.11 to leverage wider prebuilt wheels (fewer compiles)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only dependency files first (for caching)
COPY requirements.txt .

# Upgrade pip tooling and install Python deps
RUN pip install --upgrade pip setuptools wheel \
	&& pip install -r requirements.txt

# Pre-download the sentence-transformers model during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy the rest of the code
COPY . .

# Expose port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
