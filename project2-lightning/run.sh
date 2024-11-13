#!/bin/bash

# Start MySQL server in the background and log output to help diagnose issues
mysqld_safe > /var/log/mysql_error.log 2>&1 &

echo "Starting MySQL server..."

# Wait until MySQL server is ready
while ! mysqladmin ping -h 127.0.0.1 --silent; do
    echo "Waiting for MySQL to be fully initialized..."
    tail -n 10 /var/log/mysql_error.log  # Show the last 10 lines of the error log
    sleep 2
done

#mysql -u root -e "GRANT ALL PRIVILEGES ON *.* TO 'root'@'localhost' IDENTIFIED BY 'rootpassword'; FLUSH PRIVILEGES;"
# Create the 'example' database if it does not already exist
mysql -u root -e "CREATE DATABASE IF NOT EXISTS example;"

optuna create-study --study-name "distributed-example" --storage "mysql://root@localhost/example" || echo "Study already exists."

exec python3 src/train.py run1 &
exec python3 src/train.py run2