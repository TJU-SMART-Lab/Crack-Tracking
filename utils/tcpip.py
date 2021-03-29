import socket
import json
import time


def tcpIp(x, y):
    s = socket.socket()
    s.connect(('127.0.0.1', 8800))  # 连接服务端
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
    s.send(json.dumps(msg).encode())  # 发送数据
    time.sleep(0.1)
    s.close()


if __name__ == '__main__':
    tcpIp(55, 5)

"""
{
  "task": {
    "name": "test",
    "param": {
      "_speed": 1.0,
      "_target1": 1.0,
      "_target2": 1.0
    },
      "cmd": "run"
    }
}
"""
