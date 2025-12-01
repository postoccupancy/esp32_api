import os

name = os.getenv("USER_NAME", "Guest")
print(f"Hello, {name}!")