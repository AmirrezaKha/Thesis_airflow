# Master Thesis Project

This project is part of my master's thesis and involves processing data provided in CSV files. The implementation utilizes Apache Airflow for workflow orchestration and PostgreSQL for data storage, all containerized using Docker. With the help of Airflow ass well as Docker, this code has the ability to create and save several tables in proper ways that can be used for aaplying Machine Learning models as well as data visualization.

## Project Overview

- **Data Source:** CSV files
- **Workflow Orchestration:** Apache Airflow
- **Database:** PostgreSQL
- **Containerization:** Docker
- **Visualization:** Matplotlib & Power BI

## How to Run

1. **Clone the repository:**
   ```sh
   git clone https://github.com/AmirrezaKha/Thesis_airflow.git
   cd Thesis_airflow

2. **Security:**

For security purposes, usernames and passwords for these applications are managed through environment variables, and the `.env` file needs to be defined.

These variables are defined in the `.env` file:

- `POSTGRES_USER=`
- `POSTGRES_PASSWORD=`
- `POSTGRES_DB=`
- `AIRFLOW_IMAGE_NAME=extending_airflow`
- `AIRFLOW_UID=`
- `AIRFLOW_GID=`
- `_AIRFLOW_WWW_USER_USERNAME=`
- `_AIRFLOW_WWW_USER_PASSWORD=`

3. **Install the required Python libraries:**

Create a requirements.txt file with the following content:

- `statsmodels==0.13.2`
- `scikit-learn==1.0.2`
- `pandas==1.3.5`
- `numpy>=1.22,<1.24.4`
- `matplotlib==3.5.2`
- `seaborn==0.11.2`
- `tensorflow==2.8.0`
- `protobuf==3.20.3`

4. **Dockerfile setup:**
      ```sh
   FROM apache/airflow:2.7.0

   # Upgrade pip
   RUN pip install --upgrade pip

   # Copy and install requirements
   COPY requirements.txt /requirements.txt
   RUN pip install -r /requirements.txt
5. **Build and run the Docker containers:**
   Ensure Docker is installed and running on your system. Then, execute:
      ```bash
      docker-compose up -d
6. **Access Apache Airflow:**

Open your web browser and go to [http://localhost:8080.](http://localhost:8080.) Log in with the credentials you defined in the .env file.

## Additional Information

 - Ensure that your Docker and Docker Compose are properly installed and configured.
 - Follow best practices for managing your environment variables and secrets.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


This document provides a comprehensive overview and setup instructions for Master Thesis Project, ensuring that all necessary details are included.


