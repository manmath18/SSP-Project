from fastapi import FastAPI
import subprocess
import os

app = FastAPI()

@app.get("/")
def run_streamlit():
    # Run Streamlit as a subprocess
    cmd = ["streamlit", "run", "app.py", "--server.port", "8000", "--server.headless", "true", "--browser.serverAddress", "0.0.0.0"]
    subprocess.Popen(cmd, env=os.environ)
    return {"message": "Streamlit app is running!"}
