# CTF_Reverse，题目：offShitft-2021-ware
flag{32117406899806798980909}

The binary is UPX-packed. After unpacking it is a Go binary that “encrypts” a plaintext with AES and writes a decoy ciphertext to encryptedmessage.txt.
The challenge description states the plaintext consists of numbers with a space that must be removed. In main.Encryptfinal the string loaded immediately before the call to main.EncryptAES is exactly
text321174068998067 98980909

Removing the space yields the required plaintext, which is the flag.
