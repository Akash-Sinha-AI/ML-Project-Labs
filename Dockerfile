FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

# Install RunPod SDK and ML lab libraries
RUN pip install --no-cache-dir \
    runpod \
    transformers \
    sentencepiece \
    accelerate \
    scipy \
    scikit-learn

# Copy worker script
COPY handler.py /workspace/handler.py

# Unbuffer Python stdout for real-time logs
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "/workspace/handler.py"]
