"""Who can read what.

A document's `access` field names its level ('all' or 'hr'). A role grants a set of levels.
Fail closed: an unknown role gets the most restrictive access, and a document without a level is treated as 'hr'.
"""

LEVELS = {
    "visitor": ("all",),
    "employee": ("all",),
    "hr": ("all", "hr"),
}


def role_for(logged_in, email, hr_emails):
    """visitor when not logged in, hr when the email is on the HR list, employee otherwise."""
    if not logged_in:
        return "visitor"
    hr = {e.lower() for e in hr_emails}
    return "hr" if (email or "").lower() in hr else "employee"


def levels_for(role):
    return LEVELS.get(role, LEVELS["visitor"])
