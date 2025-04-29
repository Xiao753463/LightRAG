import os
from dotenv import load_dotenv

load_dotenv()  # 載入 .env

print("NEO4J_URI:", os.getenv("NEO4J_URI"))
print("NEO4J_USERNAME:", os.getenv("NEO4J_USERNAME"))
print("NEO4J_PASSWORD:", os.getenv("NEO4J_PASSWORD"))
