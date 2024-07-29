FROM apache/airflow:2.7.0

# Upgrade pip
RUN pip install --upgrade pip

# Copy and install requirements
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt