# Power BI Report Overview

This PDF file contains the results of a Power BI analysis connected to PostgreSQL. The report includes data from four different tables: `a500`, `mining table`, `classification`, and `regression` tables. It provides descriptive analysis, insightful time series plots, and results from various machine learning models.

## View the Report

You can view the Power BI report by downloading the PDF.

[View PDF](https://github.com/AmirrezaKha/Thesis_airflow/blob/master/power%20bi/PBI.pdf)

## Connecting PostgreSQL to Power BI

To connect Power BI to a PostgreSQL database running in a Docker container, follow these steps:

1. **Ensure PostgreSQL is Running in Docker:**
   Make sure you have a running PostgreSQL container. You can use the following command to run PostgreSQL in a Docker container:
   ```sh
   docker run --name my_postgres -e POSTGRES_PASSWORD=mysecretpassword -d postgres

2. **Open Power BI:**
   Launch Power BI Desktop on your machine.
3. **Enter Connection Information:**
   - Server: Enter the IP address of your PostgreSQL container followed by the port number (usually 5432). For example: 192.168.99.100:5432 or localhost simply.
   - Database: Enter the name of your database.
   - User name: Enter the PostgreSQL username.
   - Password: Enter the PostgreSQL password you specified (in this example, mysecretpassword).

4. **Select Tables:**
   After entering the connection details, click "OK" and select tables from view column. Click on  Load after choosing all tables.
