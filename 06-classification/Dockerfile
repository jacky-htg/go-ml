FROM ubuntu
WORKDIR /app
COPY ./api /app/
COPY ./iris-training.csv /app/
COPY ./html /app/html/
EXPOSE 9111:9111
CMD [ "/app/api"]