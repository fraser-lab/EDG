from utils.register_api import register_api
from utils.key_location import (
    KEY_LOCATION,
)  # Define your own key_location.py file with the path to your Chroma API key as KEY_LOCATION

def main():
    register_api(KEY_LOCATION)


if __name__ == "__main__":
    main()
