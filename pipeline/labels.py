def game_label(opponent: str, venue: str) -> str:
    """'vs. Linfield (H)' for home games, 'at Linfield (A)' for away games."""
    v = venue.strip().upper()
    if v == "H":
        return f"vs. {opponent} (H)"
    if v == "A":
        return f"at {opponent} (A)"
    raise ValueError("venue must be 'H' (home) or 'A' (away)")
