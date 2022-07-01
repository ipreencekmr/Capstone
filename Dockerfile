# Create the base image
FROM python:3.9-slim

# Change the working directory
WORKDIR /app/

# Install Dependency
COPY requirements.txt /app/
RUN pip install -r ./requirements.txt
RUN mkdir -p /app/uploads

# Copy local folder into the container
COPY app.py /app/
COPY gender_model.pkl /app/
COPY age_model.pkl /app/
COPY model.py /app/
COPY scaler.gz /app/
COPY test.csv /app/
COPY templates/index.html /app/templates/index.html


# Set "python" as the entry point
ENTRYPOINT ["python"]

# Set the command as the script name
CMD ["app.py"]

#Expose the post 5000.
EXPOSE 5000