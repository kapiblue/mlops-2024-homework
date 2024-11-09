FROM python:3.11

WORKDIR /app

COPY ./data ./data

# Upgrade pip
RUN python -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt

COPY ./src ./src

CMD [ "python", "src/train.py"]