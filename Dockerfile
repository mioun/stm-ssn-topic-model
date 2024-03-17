FROM python:3.9.6

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONIOENCODING=utf-8

ADD requirements.txt .

RUN pip install -U pip



