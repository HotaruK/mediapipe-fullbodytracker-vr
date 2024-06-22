import socket
import struct

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
VRC_OSC_PORT = 9000


def send_osc_message(address, values, port=VRC_OSC_PORT):
    msg = struct.pack('>s', address.encode())
    for value in values:
        msg += struct.pack('>f', value)
    sock.sendto(msg, ('127.0.0.1', port))
