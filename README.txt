Gesture Classification

# Requirement: Docker

------------------------------------------------------------------
1. Getting Started

First run:
Build & run docker image:

docker build -t gesture_classification . && docker run -it --init -p 8000:8000 -p 8001:8001 gesture_classification

This could take more than 3 minutes to build.
(This will also download and extract the data in the container)


Subsequent runs:
Simply run a container with the 'gesture_classification' image:

docker run -it --init -p 8000:8000 -p 8001:8001 gesture_classification


------------------------------------------------------------------
2. Usage

(Please ignore 'Port 8001 is already in use' message.)

Navigate to Swagger-UI to interact with endpoints conveniently
http://0.0.0.0:8000/swagger-ui/


Monitor models at
http://0.0.0.0:8001

------------------------------------------------------------------
3. Misc.

Exploratory data analysis in <eda.html> & <eda.ipynb>
