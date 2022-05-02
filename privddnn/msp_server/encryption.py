from Cryptodome.Cipher import AES


AES_BLOCK_SIZE = 16


def decrypt_aes128(ciphertext: bytes, key: bytes) -> bytes:
    """
    Decrypts the given cipher text using the AES-128 block cipher.

    Args:
        ciphertext: The cipher text to decrypt
        key: The symmetric encryption key
    Returns:
        The plaintext message
    """
    assert len(key) == AES_BLOCK_SIZE, 'Must provide a {}-byte key. Got {} bytes.'.format(AES_BLOCK_SIZE, len(key))

    # Extract the IV and message components
    iv = ciphertext[:AES_BLOCK_SIZE]
    ciphertext = ciphertext[AES_BLOCK_SIZE:]

    # Decrypt the message
    cipher = AES.new(key, AES.MODE_CBC, iv)
    message = cipher.decrypt(ciphertext)

    return message
