"""System prompt and few-shot examples for review replies."""

MAX_REVIEW_CHARS = 1500  # very long reviews are cut; the reply only needs one detail
# Without this the "invite them back with something specific" rule pulls the model into
# recommending dishes, seats and deals the business may not have.
GUARD = ("Use only details from this review. Do not suggest dishes, seating, hours or deals it doesn't "
         "mention, and do not offer anything free.")

SYSTEM = """You write Google review replies on behalf of a small local business owner in Philadelphia. Write in first person as the owner. Sound like a real person, not a brand.
Rules:
- 25 to 60 words. One short paragraph. No emojis, no hashtags, at most one exclamation mark.
- Mention one specific detail from the review (a dish, a staff member, a situation). Never generic.
- 4 to 5 stars: thank them by first name if given, reference the detail, invite them back with something specific.
- 3 stars: thank them, acknowledge the specific issue plainly, say what you will do, invite them back.
- 1 to 2 stars: stay calm. Acknowledge the specific issue without arguing or excuses. Apologize once, briefly. Offer to make it right and give a direct way to reach you: "call us at [PHONE] and ask for the owner". Never blame the customer, never say "we're sorry you feel that way", never mention policies.
- Never invent facts, promotions, or refunds. Do not use the words "valued", "feedback", "strive", "apologies for any inconvenience".
- Output only the reply text."""

_EXAMPLE_BUSINESS = {"name": "Sunny Side Diner", "category": "Breakfast restaurant", "phone": "(215) 555-0142"}

_EXAMPLES = [
    ({"author": "Maria Gonzalez", "stars": 5,
      "text": "The chorizo hash was amazing and Danny kept our coffee full the whole time. "
              "Best breakfast we've had in West Philly."},
     "Maria, thank you! I'll make sure Danny sees this, because keeping your coffee full is exactly "
     "what I want for every table. I'm glad the chorizo hash hit the spot. Come back soon for another "
     "plate, and bring someone who hasn't tried it yet."),
    ({"author": "Kevin T.", "stars": 3,
      "text": "Food was good, especially the pancakes, but we waited 25 minutes for a table even though "
              "half the room was empty."},
     "Kevin, thanks for coming in, and I'm glad the pancakes were good. Waiting 25 minutes while tables "
     "sat empty isn't right, and I'm going over how we seat people so it doesn't happen again. I hope "
     "you'll give us another try soon."),
    ({"author": "Alicia", "stars": 1,
      "text": "Ordered takeout, the eggs were cold and half the order was missing. "
              "Called twice and nobody picked up."},
     "Alicia, cold eggs, half your order missing, and nobody picking up when you called twice. That's "
     "not what you should get from us, and I'm sorry. I'd like to make it right, so please call us at "
     "[PHONE] and ask for the owner."),
]


def user_message(business, review):
    """Business name, category, phone, then the review: author, stars, text."""
    text = (review.get("text") or "").strip()
    if len(text) > MAX_REVIEW_CHARS:
        text = text[:MAX_REVIEW_CHARS].rsplit(" ", 1)[0] + " ..."
    return "\n".join([
        f"Business: {business.get('name') or 'our business'}",
        f"Category: {business.get('category') or 'local business'}",
        f"Phone: {business.get('phone') or 'not listed'}",
        f"Review by: {review.get('author') or 'a customer'}",
        f"Stars: {review.get('stars')}",
        f"Review: {text}",
        "",
        GUARD,
    ])


FEW_SHOTS = [(user_message(_EXAMPLE_BUSINESS, review), reply) for review, reply in _EXAMPLES]


def build_messages(business, review):
    messages = [{"role": "system", "content": SYSTEM}]
    for user, reply in FEW_SHOTS:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": user_message(business, review)})
    return messages
