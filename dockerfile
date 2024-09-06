#importing python image
FROM python:3.9-slim_buster

#dir will be creted  inside the container
WORKDIR /flask_docker

# to upgrade python
RUN python3 -m pip install --upgrade pip 

#copying local req to container
COPY requirements.txt requirements.txt

#instaling req
RUN pip3 install -r requirements.txt

# copying everything from our local to the container
COPY . .

# for running thr bult image
CMD ["python","-m","flask","--app","predict.py","run","--host=0.0.0.0"]
