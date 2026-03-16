FROM python:3.10.6-slim-bullseye
RUN apt-get update && apt-get install -y --no-install-recommends git ssh && rm -rf /var/lib/apt/lists/*
RUN mkdir /dopf_federate
COPY . /dopf_federate
WORKDIR /dopf_federate
RUN pip install --no-cache-dir -e .
EXPOSE 5683/tcp
CMD ["python", "-m", "dopf_federate.server"]
