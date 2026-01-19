# Base image
FROM python:3.11-slim AS base-image

#===================================================================================================
# Base module builder
#===================================================================================================
FROM base-image AS builder

# Install the package dependencies
COPY wheels/ /wheels/
COPY requirements.txt /app/
RUN python -m venv /app/venv \
	&& ./app/venv/bin/python -m pip install --no-cache-dir -U pip \
	&& ./app/venv/bin/pip install --no-cache-dir -f /wheels -r /app/requirements.txt

# Install the package
COPY MANIFEST.in pyproject.toml setup.cfg setup.py /app/
COPY src/ /app/src/
ARG GIT_DESCRIBE_STRING
RUN cd /app/ \
	&& ./venv/bin/pip install --no-cache-dir .

#===================================================================================================
# Production image
#===================================================================================================
FROM base-image

RUN useradd worker

COPY --from=builder /app/venv/ /app/venv/

USER worker

ENTRYPOINT ["/app/venv/bin/aerial-flower-intensity"]
