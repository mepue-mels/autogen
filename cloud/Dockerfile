FROM python:3.12


ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

EXPOSE 8080

RUN pip3 install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


CMD exec gunicorn --bind :$PORT --workers 4 --threads 8 --timeout 0 main:app
