FROM python:3.9

WORKDIR /usr/src/app

# Copy the current directory contents into the container
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
