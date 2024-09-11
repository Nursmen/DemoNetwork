from composio.cli.context import get_context
from composio.cli.logout import _logout
from composio.cli.login import _login
from composio.utils.url import get_web_url
from composio.client import Composio
from composio.cli.context import Context

def logout():
    context = get_context()

    user_data = context.user_data
    user_data.api_key = None
    user_data.store()

def login():
    context = get_context()

    key = Composio.generate_auth_key()
    # key = "6e96e619-3abb-4ab3-99d7-15260df0107d"
    url = get_web_url(path=f"?cliKey={key}")

    return url, key
    
def authentificate(code:str, key:str) -> bool:
    context = get_context()
    if context is None:
        context = Context()

    try:
        api_key = Composio.validate_auth_session(
            key=key,
            code=code,
        )
    except:
        print("Authentification failed!")
        return False

    context.user_data.api_key = api_key
    context.user_data.store()
    print("Authentification successful!")

    return True


if __name__ == "__main__":
    logout()
    url, key = login()
    print(url)

    authentificate(input(), key)