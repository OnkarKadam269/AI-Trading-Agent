import os
import json
from cryptography.fernet import Fernet

def encrypt_and_save():
    # Credentials to encrypt
    username = "OnkarKadam269"
    pat = "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"

    # Generate an encryption key
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)

    # Encrypt the data
    encrypted_username = cipher_suite.encrypt(username.encode()).decode()
    encrypted_pat = cipher_suite.encrypt(pat.encode()).decode()

    # Save the encrypted data
    encrypted_data = {
        "GITHUB_USERNAME": encrypted_username,
        "GITHUB_PAT": encrypted_pat
    }
    
    with open("encrypted_secrets.json", "w") as f:
        json.dump(encrypted_data, f, indent=4)
        
    # Save the decryption key (Must be kept completely safe)
    with open("decryption.key", "wb") as f:
        f.write(key)

    print("Credentials successfully encrypted and saved to 'encrypted_secrets.json'.")
    print("The decryption key has been saved to 'decryption.key'.")

if __name__ == "__main__":
    encrypt_and_save()
