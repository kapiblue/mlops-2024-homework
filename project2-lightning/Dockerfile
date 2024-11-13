FROM mysql:8.0-debian

WORKDIR /app

COPY ./data ./data

ENV PYTHON_VERSION 3.11.0

# Install python3 and pip
RUN apt-get update \
  && apt-get install -y python3 python3-pip \  
  && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip --break-system-packages

COPY requirements.txt .
RUN pip3 install -r ./requirements.txt --break-system-packages

COPY ./src ./src

COPY ./run.sh .
RUN chmod a+x run.sh

ENV MYSQL_ROOT_PASSWORD=rootpassword
ENV MYSQL_DATABASE=example
EXPOSE 3306

CMD ["./run.sh"]

# CMD [ "python3", "src/train.py"]