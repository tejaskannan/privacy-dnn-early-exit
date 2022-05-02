from collections import namedtuple
from .encryption import decrypt_aes128


Response = namedtuple('Response', ['prediction', 'exit_decision'])


def decode_message(message: bytes, key: bytes) -> Response:
    # Decrypt the message
    plaintext = decrypt_aes_128(ciphertext=message, key=key)

    # Return the prediction (first byte of the result)
    prediction = int.from_bytes(message[0], byteorder='big')
    decision = int.from_bytes(message[1], byteorder='big')

    return Response(prediction=prediction, exit_decision=decision)
