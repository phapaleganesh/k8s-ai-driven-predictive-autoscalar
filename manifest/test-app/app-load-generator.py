import requests
import threading
import time

TARGET_URL = "http://<your-app-service>:<port>"  # Replace with your app's service endpoint

def load():
    while True:
        try:
            requests.get(TARGET_URL)
        except Exception as e:
            print(e)
        time.sleep(0.1)  # Adjust for more/less load

threads = []
for _ in range(20):  # Number of concurrent threads
    t = threading.Thread(target=load)
    t.daemon = True
    t.start()
    threads.append(t)

time.sleep(600)  # Run for 10 minutes

