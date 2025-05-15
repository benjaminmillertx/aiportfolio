# telemetry_server.py - Harvard-Level AI Drone Component
# Author: Benjamin Miller (Ethical AI & Security Design)

import socket
import threading
import json

class TelemetryServer:
    def __init__(self, host='0.0.0.0', port=8888):
        self.host = host
        self.port = port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.authenticated_clients = []

    def start_server(self):
        self.server.bind((self.host, self.port))
        self.server.listen(5)
        print(f"[+] Secure Telemetry Server running on {self.host}:{self.port}")
        while True:
            client, addr = self.server.accept()
            print(f"[+] Connection from {addr}")
            threading.Thread(target=self.handle_client, args=(client, addr)).start()

    def handle_client(self, client, addr):
        try:
            data = client.recv(1024).decode()
            creds = json.loads(data)
            if self.authenticate(creds['username'], creds['password']):
                self.authenticated_clients.append(addr)
                client.send(b"AUTH_SUCCESS")
            else:
                client.send(b"AUTH_FAIL")
                client.close()
        except Exception as e:
            print(f"[!] Error with {addr}: {e}")
            client.close()

    def authenticate(self, username, password):
        # This function can be tested with bruteforce tools like Hydra in an ethical CTF setup
        return username == "admin" and password == "securepass123"

if __name__ == "__main__":
    server = TelemetryServer()
    server.start_server()
