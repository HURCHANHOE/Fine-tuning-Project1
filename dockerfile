FROM silverlogic/python3.8

RUN mkdir ~/.pip
RUN mkdir ~/.config
RUN mkdir ~/.config/pip

# python code copy
COPY ./summarization /summarization
COPY ./requirements.txt /requirements.txt

# install python package
RUN pip install -r ./requirements.txt --default-timeout=10000

WORKDIR /summarization/ktnlp/common

