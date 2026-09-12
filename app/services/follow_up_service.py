from app.schemas.research import (
    FollowUpQuestion,
    ResearchPurpose,
)


FOLLOW_UP_QUESTIONS: dict[
    ResearchPurpose,
    list[FollowUpQuestion],
] = {
    ResearchPurpose.GENERAL_RESEARCH: [
        FollowUpQuestion(
            id="target_location",
            question="Which country or city should we focus on?",
            options=["Worldwide", "India", "Other"],
            placeholder="Example: Chennai",
        ),
        FollowUpQuestion(
            id="product_focus",
            question="Is there a specific product or service you want us to focus on?",
            placeholder="Example: Beverages",
        ),
        FollowUpQuestion(
            id="priority_area",
            question="What would you like to understand most?",
            options=[
                "Customers",
                "Employees",
                "Finances",
                "Competitors",
                "Everything",
            ],
        ),
    ],

    ResearchPurpose.START_SIMILAR_BUSINESS: [
        FollowUpQuestion(
            id="target_location",
            question="Where are you planning to start this business?",
            placeholder="Example: Chennai",
        ),
        FollowUpQuestion(
            id="investment_amount",
            question="How much are you planning to invest approximately?",
            options=[
                "Below ₹5 lakh",
                "₹5–20 lakh",
                "₹20–50 lakh",
                "Above ₹50 lakh",
                "Not decided",
            ],
        ),
        FollowUpQuestion(
            id="business_type",
            question="How will your business operate?",
            options=[
                "Online",
                "Physical location",
                "Both",
                "Not decided",
            ],
        ),
    ],

    ResearchPurpose.PARTNER_OR_SUPPLIER: [
        FollowUpQuestion(
            id="offering",
            question="What product or service would you like to offer this company?",
            placeholder="Describe your product or service",
        ),
        FollowUpQuestion(
            id="target_location",
            question="Where would you like to work with this company?",
            placeholder="Example: South India",
        ),
        FollowUpQuestion(
            id="relationship_type",
            question="How would you like to work with this company?",
            options=[
                "Supplier",
                "Distributor",
                "Technology provider",
                "Business partner",
                "Not sure",
            ],
        ),
    ],

    ResearchPurpose.STOCK_RESEARCH: [
        FollowUpQuestion(
            id="holding_period",
            question="How long are you planning to hold this investment?",
            options=[
                "Less than 1 year",
                "1–3 years",
                "More than 3 years",
                "Not decided",
            ],
        ),
        FollowUpQuestion(
            id="risk_level",
            question="How much investment risk are you comfortable with?",
            options=[
                "Low",
                "Medium",
                "High",
                "Not sure",
            ],
        ),
        FollowUpQuestion(
            id="investment_priority",
            question="What matters most to you?",
            options=[
                "Company growth",
                "Regular dividends",
                "Financial stability",
                "Current share price",
                "Everything",
            ],
        ),
    ],
}


def get_follow_up_questions(
    purpose: ResearchPurpose,
) -> list[FollowUpQuestion]:
    return FOLLOW_UP_QUESTIONS.get(purpose, [])