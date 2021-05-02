FROM ubuntu:20.04

RUN apt-get update -y && \
    apt-get install -y python3-pip python-dev vim curl

RUN pip3 install --upgrade pip

COPY ./app /app
COPY requirements.txt /tmp/

WORKDIR /app

RUN pip3 install -r /tmp/requirements.txt -t /app
RUN pip3 install gevent -t /app --upgrade

ENTRYPOINT [ "python3" ]

CMD ["main.py"]
