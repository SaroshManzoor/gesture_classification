FROM python:3.9.1 as requirements-stage

WORKDIR /tmp


RUN pip install poetry -U
COPY ./pyproject.docker.toml /tmp/pyproject.toml
# Not copying the lock file because of tensorflow-macos discrepency
# The two stage approach will take longer to run but will run without issues
# regarding virtual envs
RUN poetry install --without dev

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

FROM python:3.9.1

WORKDIR /src
COPY --from=requirements-stage /tmp/requirements.txt /src/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /src/requirements.txt

COPY ./download_and_extract_data_docker.sh /src/download_and_extract_data_docker.sh
COPY ./gesture_classification /src/gesture_classification
COPY ./docker-entrypoint.sh /src/docker-entrypoint.sh

RUN apt-get update
RUN apt-get install unrar-free -y
RUN apt-get install unzip -y


RUN chmod +x /src/download_and_extract_data_docker.sh
RUN /src/download_and_extract_data_docker.sh

EXPOSE 8000
ENTRYPOINT exec ./docker-entrypoint.sh