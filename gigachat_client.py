import os
from gigachat import GigaChat
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_gigachat_client():
    """
    Returns an authenticated GigaChat client.
    """
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    verify_ssl = os.getenv("GIGACHAT_VERIFY_SSL", "True").lower() == "true"
    
    if not credentials:
        raise ValueError("GIGACHAT_CREDENTIALS not found in environment variables.")

    chat = GigaChat(
        credentials=credentials,
        verify_ssl_certs=verify_ssl,
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
    )
    return chat

def test_connection():
    """
    Tests the connection to GigaChat API.
    """
    try:
        print("Connecting to GigaChat...")
        chat = get_gigachat_client()
        response = chat.chat("Привет! Ты работаешь?")
        print(f"Response: {response.choices[0].message.content}")
        print("Connection successful!")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()
