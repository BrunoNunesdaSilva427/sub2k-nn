#!/usr/bin/env python3

import sys
import time
import argparse
import numpy as np

SYNC_BYTE = 0xAA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--index", type=int, default=None, help="índice específico do dataset (senão, aleatório)")
    args = parser.parse_args()

    try:
        import serial
    except ImportError:
        print("pyserial não instalado. Rode: pip install pyserial --break-system-packages", file=sys.stderr)
        sys.exit(1)

    from sklearn.datasets import load_digits
    digits = load_digits()

    idx = args.index if args.index is not None else np.random.randint(len(digits.data))
    pixels = digits.data[idx].round().astype(np.uint8)
    true_label = int(digits.target[idx])

    checksum = int(pixels.sum()) & 0xFF
    packet = bytes([SYNC_BYTE]) + pixels.tobytes() + bytes([checksum])

    print(f"amostra #{idx}  (rótulo real: {true_label})")
    print(f"conectando em {args.port} @ {args.baud} baud...")

    with serial.Serial(args.port, args.baud, timeout=3) as ser:
        time.sleep(2)  
        ser.reset_input_buffer()
        ser.write(packet)
        response = ser.readline().decode(errors="replace").strip()
        print(f"resposta do Arduino: {response}")

        if response.startswith("PRED="):
            pred = int(response.split()[0].split("=")[1])
            status = "ACERTOU" if pred == true_label else "ERROU"
            print(f"-> {status} (previsto={pred}, real={true_label})")


if __name__ == "__main__":
    main()
