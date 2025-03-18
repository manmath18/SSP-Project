from fastapi import FastAPI
import subprocess

app = FastAPI()

@app.get("/")
def run_streamlit():
    subprocess.run(["streamlit", "run", "app.py", "--server.port", "8000", "--server.headless", "true"])
    return {"message": "Streamlit app is running"}
