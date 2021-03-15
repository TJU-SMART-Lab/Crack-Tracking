import socket


def tcpIp(x, y):
    s = socket.socket()
    s.connect(('127.0.0.1', 9999))      # 连接服务端
    s.send(x)       # 发送数据
    s.send(y)
    s.close()
