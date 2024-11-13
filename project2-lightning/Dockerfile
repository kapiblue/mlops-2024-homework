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

# Install requirements
COPY requirements.txt .
RUN pip3 install -r ./requirements.txt --break-system-packages

# Install mysqlclient with dependencies
RUN apt-get update && apt-get install -y pkg-config python3-dev default-libmysqlclient-dev build-essential \
    default-libmysqlclient-dev && pip3 install mysqlclient --break-system-packages


#ENV MYSQL_ROOT_PASSWORD=rootpassword
RUN pip3 install optuna-integration[pytorch_lightning] --break-system-packages

# Force Data Directory Initialization
RUN chown -R mysql:mysql /var/lib/mysql /var/run/mysqld && \
chmod -R 755 /var/lib/mysql /var/run/mysqld
RUN mysqld --initialize-insecure --user=mysql --datadir=/var/lib/mysql

#ENV MYSQL_ALLOW_EMPTY_PASSWORD=yes
ENV MYSQL_ROOT_PASSWORD=rootpassword
ENV MYSQL_DATABASE=example
EXPOSE 3306

COPY ./src ./src
COPY ./run.sh .
RUN chmod a+x run.sh

ENTRYPOINT ["./run.sh"]