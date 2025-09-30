import requests
import threading
import time

TARGET_URL = "http://localhost:8089/"  # Replace with your app's service endpoint

def load():
    while True:
        try:
            requests.get(TARGET_URL)
        except Exception as e:
            print(e)
        time.sleep(0.5)  # Adjust for more/less load

threads = []
for _ in range(30):  # Number of concurrent threads
    t = threading.Thread(target=load)
    t.daemon = True
    t.start()
    threads.append(t)

time.sleep(1200)  # Run for 10 minutes

