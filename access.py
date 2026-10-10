"""Who can read what.

A document's `acesso` field names its level ('todos' or 'rh'). A role grants a set of levels.
Fail closed: an unknown role gets the most restrictive access, and a document without a level is treated as 'rh'.
"""

LEVELS = {
    "visitante": ("todos",),
    "colaborador": ("todos",),
    "rh": ("todos", "rh"),
}


def role_for(logged_in, email, rh_emails):
    """visitante when not logged in, rh when the email is on the HR list, colaborador otherwise."""
    if not logged_in:
        return "visitante"
    hr = {e.lower() for e in rh_emails}
    return "rh" if (email or "").lower() in hr else "colaborador"


def levels_for(role):
    return LEVELS.get(role, LEVELS["visitante"])
