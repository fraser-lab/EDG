from chroma import api

def register_api(key_location: str) -> None:
    """Register your Chroma API key.

    Parameters
    ----------
    key_location : str
        Path to a file containing your Chroma API key.
    """
    with open(key_location, "r") as f:
        api_key = f.read().strip()

    api.register_key(api_key)