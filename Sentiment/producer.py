import socket
import time
import json
import random

HOST = "localhost"
PORT = 9999

tweets = [
    {"username": "alice", "content": "I love AI, it’s amazing!"},
    {"username": "bob", "content": "Spark streaming is tricky."},
    {"username": "charlie", "content": "This is so frustrating..."},
    {"username": "dave", "content": "I am very happy today!"}
]

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

while True:
    tweet = random.choice(tweets)
    msg = json.dumps(tweet) + "\n"   # newline is important
    s.sendall(msg.encode("utf-8"))
    print("Sent:", msg.strip())
    time.sleep(2)
