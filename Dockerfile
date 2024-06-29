FROM docker.arvancloud.ir/python:latest

WORKDIR /code

COPY requirements.txt /code/

RUN pip install -U pip && \
    pip install -r requirements.txt

COPY . /code/

CMD ["sh", "-c", "python main.pyr"]
