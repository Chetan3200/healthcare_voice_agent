"""Session-only case selection; the five clinical read contracts stay unchanged."""
from typing import Annotated, Literal, Union

from pydantic import Field, StrictStr, model_validator
from healthcare_voice_agent.tools.contracts import (
    ContractModel, ErrorResponse, RecordId, SuccessResponse, ToolContract,
)


class OpenCaseArgs(ContractModel):
    case_id: RecordId | None = Field(description=(
        "Four-digit case ID normalized from the caller's spoken number, e.g. 'ten forty-two' "
        "means '1042', or JSON null for a name search. Preserve explicit leading zeros; never "
        "guess missing digits. Clear spoken numbers need no extra confirmation. "
        'Never send the strings "None" or "null".'))
    patient_name: Annotated[StrictStr, Field(min_length=1, max_length=120)] | None = Field(
        description='Patient name/name fragment for identity candidates only, or JSON null when opening an exact case ID. Never send the strings "None" or "null".')

    @model_validator(mode="after")
    def one_identifier(self):
        if (self.case_id is None) == (self.patient_name is None):
            raise ValueError("Supply either case_id or patient_name, with the other explicitly null.")
        if not (self.case_id or self.patient_name).strip():
            raise ValueError("An identifier cannot be blank.")
        return self


class CaseIdentity(ContractModel):
    case_id: RecordId
    patient_id: RecordId
    patient_display_name: StrictStr


class OpenCaseData(ContractModel):
    state: Literal["selected", "selection_required"]
    selected_case: CaseIdentity | None
    candidates: Annotated[list[CaseIdentity], Field(max_length=50)]
    has_more: bool


OpenCaseResponse = Annotated[Union[
    SuccessResponse[OpenCaseData, Literal["open_case"]],
    ErrorResponse[Literal["open_case"]],
], Field(discriminator="status")]

OPEN_CASE = ToolContract(
    "open_case",
    "Open the caller's four-digit numeric case ID, normalizing clear spoken numbers "
    "without extra confirmation, or find minimal authorized identity candidates by name. Name matches return state=selection_required, NOT an "
    "opened case, even when status=ok. Offer the returned case IDs and ask which one; "
    "never select automatically or retrieve candidate clinical records to distinguish them. "
    "Do not disclose or enumerate birth dates, or require DOB verification for an exact case ID. "
    "Similar names are not identity proof. "
    "This clears the previous selected case even on failure/ambiguity; never fall back "
    "to old patient records. Call this before the four clinical read tools and after "
    "an identifier correction. Wait for status=ok AND data.state=selected in a completed "
    "tool round before clinical reads; never batch those reads with open_case. "
    "Independent reads may run in parallel afterwards. No database writes.",
    OpenCaseArgs, OpenCaseResponse, read_only=False,
)
