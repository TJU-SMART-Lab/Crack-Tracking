import socket
import json
import time


def tcpIp(x, y):
    msg = {
        "task": {
            "name": "test",
            "param": {
                "_speed": 1.0,
                "_target1": x,
                "_target2": y
            },
            "cmd": "run"
        }
    }
    print(msg)
    try:
        s = socket.socket()
        s.connect(('127.0.0.1', 8800))  # 连接服务端
        s.send(json.dumps(msg).encode())  # 发送数据
        time.sleep(0.5)
        s.close()
    except ConnectionRefusedError:
        print('端口错误')


if __name__ == '__main__':
    tcpIp(55, 5)
