# Initialization of the project
docker-compose down
docker-compose build
docker-compose up
################################
# Run postgresql in container
docker ps
docker exec -it your_postgres_container bash

psql -U your_postgres_user -d your_postgres_db

################################
# Stop All Services:
docker-compose stop

# Start All Services:

docker-compose start

# Pause All Services:

docker-compose pause

# Unpause All Services:

docker-compose unpause

# Restart All Services:

docker-compose restart

# Stop a Specific Service:

docker-compose stop db

