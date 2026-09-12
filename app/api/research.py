from fastapi import APIRouter

from app.schemas.research import (
    FollowUpQuestionsResponse,
    ResearchPurpose,
    ResearchRequest,
    ResearchResponse,
)
from app.services.follow_up_service import get_follow_up_questions


router = APIRouter(
    prefix="/api/research",
    tags=["Research"],
)


@router.get(
    "/follow-up-questions/{purpose}",
    response_model=FollowUpQuestionsResponse,
)
def get_questions(
    purpose: ResearchPurpose,
) -> FollowUpQuestionsResponse:
    return FollowUpQuestionsResponse(
        purpose=purpose,
        questions=get_follow_up_questions(purpose),
    )


@router.post(
    "",
    response_model=ResearchResponse,
)
def create_research(
    request: ResearchRequest,
) -> ResearchResponse:
    return ResearchResponse(
        status="accepted",
        company_request=request.company_request,
        purpose=request.purpose,
        official_website=request.official_website,
        report_depth=request.report_depth,
        follow_up_answers=request.follow_up_answers,
        message="Research request validated successfully",
    )