"""Serve a LeRobot policy to a simulator that lives in ANOTHER Python environment.

LIBERO-Plus pins numpy<2, mujoco 2.3.7 and robosuite 1.4.1; current LeRobot wants none of those. On one machine the
two can still work together: this process owns the policy (any Python, CUDA torch, LeRobot), the simulator process
talks to it over a local socket through `policy_adapters.RemoteAdapter`. Observations cross as plain numpy arrays,
actions come back as lists.

    <lerobot venv>/python policy_server.py --policy lerobot/smolvla_libero --port 6006
    <sim venv>/python sweep_run.py episodes --family smolvla ...      (RemoteAdapter connects to the same port)
"""
import argparse
import os
import sys
from multiprocessing.connection import Listener

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="lerobot/smolvla_libero")
    ap.add_argument("--port", type=int, default=6006)
    a = ap.parse_args()
    try:
        import truststore
        truststore.inject_into_ssl()              # machines whose TLS goes through a local proxy
    except ImportError:
        pass
    from policy_adapters import LeRobotAdapter
    adapter = LeRobotAdapter(a.policy)
    print(f"policy {a.policy} loaded on {adapter.policy.config.device}; listening on 127.0.0.1:{a.port}", flush=True)
    with Listener(("127.0.0.1", a.port), authkey=b"self-motion") as listener:
        while True:
            with listener.accept() as conn:
                print("simulator connected", flush=True)
                try:
                    while True:
                        msg = conn.recv()
                        if msg[0] == "new":
                            adapter.new_episode(msg[1], int(msg[2]))
                            conn.send("ok")
                        elif msg[0] == "chunk":
                            conn.send([[float(v) for v in act] for act in adapter.chunk(msg[1])])
                        elif msg[0] == "quit":
                            conn.send("bye")
                            return
                except (EOFError, ConnectionResetError):
                    print("simulator disconnected", flush=True)


if __name__ == "__main__":
    main()
