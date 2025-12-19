import csv
import time
import socketio
from datetime import datetime
import os
import sys

# allow reconnection attempts
sio = socketio.Client(reconnection=True, reconnection_attempts=0)

@sio.event
def connect():
    print("[simulator] Connected to server.")

@sio.event
def disconnect():
    print("[simulator] Disconnected from server.")

@sio.event
def connect_error(data):
    print("[simulator] Connection error:", data)

def wait_and_connect(url='http://localhost:8000', delay=2):
    """Try to connect until server accepts the socket connection."""
    while True:
        try:
            print(f"[simulator] Attempting to connect to {url} ...")
            sio.connect(url)
            return
        except Exception as e:
            print(f"[simulator] Connect failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)

def timestamp_to_seconds(ts_str):
    """Convert timestamp like '7/11/2025 12:00' to seconds since midnight."""
    try:
        dt = datetime.strptime(ts_str, "%m/%d/%Y %H:%M")
        return dt.hour * 3600 + dt.minute * 60
    except ValueError:
        return 0

if __name__ == "__main__":
    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, 'raw_data', 'raw_transactions.csv')
    if not os.path.exists(csv_path):
        print(f"[simulator] raw_transactions.csv not found at {csv_path}")
        sys.exit(1)

    server_url = 'http://localhost:8000'  # FastAPI port
    wait_and_connect(server_url, delay=2)

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            amount_raw = row.get('amount', '0')
            try:
                amount = float(amount_raw)
            except:
                amount = 0.0

            time_in_seconds = timestamp_to_seconds(row.get('timestamp', ''))

            transaction = {
                "amount": amount,
                "time": time_in_seconds
            }

            sio.emit('new_transaction', transaction)
            print("[simulator] Sent:", transaction)
            time.sleep(2)  # Delay between sends

    try:
        sio.disconnect()
    except:
        pass
