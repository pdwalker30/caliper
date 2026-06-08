import sqlite3


def get_user_by_username(username: str) -> tuple | None:
    """Look up a user record by username."""
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    query = "SELECT id, username, email FROM users WHERE username = '" + username + "'"
    cursor.execute(query)

    return cursor.fetchone()


def authenticate(username: str, password: str) -> bool:
    user = get_user_by_username(username)
    if user is None:
        return False
    return check_password(user, password)
