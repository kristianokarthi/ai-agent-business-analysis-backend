from app.schemas.rag import ChatMessage


def build_contextual_query(
    question: str,
    history: list[ChatMessage],
) -> str:
    """Make an ambiguous follow-up retrievable without using chat as evidence."""
    if not history:
        return question

    conversation = "\n".join(
        f"{message.role.value}: {message.content}"
        for message in history[-6:]
    )
    return (
        "Use the prior conversation only to resolve what the current question "
        "refers to.\n"
        f"Prior conversation:\n{conversation}\n"
        f"Current question: {question}"
    )
