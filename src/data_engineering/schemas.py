"""
ClaimGraph AI — Pydantic Schemas
=================================
Har entity ka ek strict schema hai — validation yahan hoti hai, baad mein nahi.
Ye schemas data generation, API, aur DB loading teeno jagah use hote hain.

Interview tip: "I used Pydantic for data validation at the boundary —
every record is validated before it enters the pipeline, so downstream
code can trust the data shape."
"""

from datetime import date, datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# === Enums — ye categories fixed hain, random strings nahi ===

class ClaimType(str, Enum):
    """Claim kis type ka hai — insurance/warranty mein common categories"""
    ACCIDENTAL_DAMAGE = "accidental_damage"
    MECHANICAL_FAILURE = "mechanical_failure"
    THEFT = "theft"
    WATER_DAMAGE = "water_damage"
    SCREEN_DAMAGE = "screen_damage"
    BATTERY_ISSUE = "battery_issue"
    SOFTWARE_ISSUE = "software_issue"


class ClaimStatus(str, Enum):
    """Claim abhi kis stage pe hai"""
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    DENIED = "denied"
    ESCALATED = "escalated"


class PolicyType(str, Enum):
    """Policy ka type — basic se premium tak"""
    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"
    EXTENDED = "extended"


class DeviceType(str, Enum):
    """Device categories — warranty claims mein common"""
    SMARTPHONE = "smartphone"
    LAPTOP = "laptop"
    TABLET = "tablet"
    SMARTWATCH = "smartwatch"
    TELEVISION = "television"
    APPLIANCE = "appliance"


class RiskLevel(str, Enum):
    """Final risk assessment — system sirf recommend karta hai, decide nahi"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class InvestigationAction(str, Enum):
    """System ki recommendation — APPROVE/DENY kabhi autonomous nahi"""
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    ESCALATE_FOR_INVESTIGATION = "ESCALATE_FOR_INVESTIGATION"


# === Entity Schemas ===

class Customer(BaseModel):
    """
    Customer record — ek insaan jo policy kharidta hai.
    customer_id unique hona chahiye, name/email required hai.
    """
    customer_id: str = Field(..., pattern=r"^CUS-\d{6}$",
                             description="Unique customer ID, format: CUS-XXXXXX")
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., pattern=r"^[\w\.\-]+@[\w\.\-]+\.\w+$")
    phone: str = Field(..., min_length=10, max_length=15)
    date_of_birth: date
    address_id: str = Field(..., pattern=r"^ADR-\d{6}$")
    payment_account_id: str = Field(..., pattern=r"^PAY-\d{6}$")
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("date_of_birth")
    @classmethod
    def dob_must_be_past(cls, v):
        """Customer ka DOB future mein nahi ho sakta"""
        if v >= date.today():
            raise ValueError("Date of birth must be in the past")
        return v


class Address(BaseModel):
    """
    Address record — fuzzy matching se duplicates detect karte hain.
    Same address = potential shared entity (fraud ya family, dono ho sakta hai).
    """
    address_id: str = Field(..., pattern=r"^ADR-\d{6}$")
    street: str = Field(..., min_length=1)
    city: str = Field(..., min_length=1)
    state: str = Field(..., min_length=2, max_length=2,
                       description="2-letter state code")
    zip_code: str = Field(..., pattern=r"^\d{5}$")
    country: str = Field(default="US")


class Device(BaseModel):
    """
    Device record — shared device across customers = strong fraud signal.
    device_id + serial_number dono track karte hain.
    """
    device_id: str = Field(..., pattern=r"^DEV-\d{6}$")
    device_type: DeviceType
    brand: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    serial_number: str = Field(..., min_length=5,
                               description="Manufacturing serial — sharing iska matlab fraud")
    purchase_date: date
    purchase_price: float = Field(..., gt=0)


class PaymentAccount(BaseModel):
    """
    Payment account — shared payment = financial link between customers.
    Agar alag alag customers same account use karein, red flag hai.
    """
    payment_account_id: str = Field(..., pattern=r"^PAY-\d{6}$")
    account_type: str = Field(..., description="bank_account / credit_card / digital_wallet")
    # Masked for security — last 4 digits only
    account_last_four: str = Field(..., pattern=r"^\d{4}$")
    bank_name: str = Field(..., min_length=1)


class RepairShop(BaseModel):
    """
    Repair shop — high-volume shops with inflated claims = fraud pattern.
    Lekin popular shops legitimately bhi bahut claims handle karti hain.
    """
    repair_shop_id: str = Field(..., pattern=r"^REP-\d{6}$")
    name: str = Field(..., min_length=1)
    address_id: str = Field(..., pattern=r"^ADR-\d{6}$")
    specialty: str = Field(..., description="electronics / appliances / general")
    rating: float = Field(..., ge=1.0, le=5.0)
    is_authorized: bool = Field(default=True,
                                description="Authorized dealer hai ya nahi")


class Policy(BaseModel):
    """
    Insurance/warranty policy — har claim ek policy se linked hota hai.
    Policy dates se time-since-policy-start feature banta hai.
    """
    policy_id: str = Field(..., pattern=r"^POL-\d{6}$")
    customer_id: str = Field(..., pattern=r"^CUS-\d{6}$")
    device_id: str = Field(..., pattern=r"^DEV-\d{6}$")
    policy_type: PolicyType
    start_date: date
    end_date: date
    premium_amount: float = Field(..., gt=0)
    coverage_limit: float = Field(..., gt=0)
    is_active: bool = Field(default=True)

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v, info):
        """Policy end date start se baad hona chahiye"""
        if "start_date" in info.data and v <= info.data["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v


class Claim(BaseModel):
    """
    Insurance claim — ye hamara main entity hai.
    Har claim ek policy, customer, device, aur optionally repair shop se linked hai.
    """
    claim_id: str = Field(..., pattern=r"^CLM-\d{6}$")
    policy_id: str = Field(..., pattern=r"^POL-\d{6}$")
    customer_id: str = Field(..., pattern=r"^CUS-\d{6}$")
    device_id: str = Field(..., pattern=r"^DEV-\d{6}$")
    repair_shop_id: Optional[str] = Field(None, pattern=r"^REP-\d{6}$")
    claim_type: ClaimType
    claim_amount: float = Field(..., gt=0)
    claim_date: date
    description: str = Field(..., min_length=10,
                             description="Claim ki detail — multimodal check isse compare karega image se")
    status: ClaimStatus = Field(default=ClaimStatus.SUBMITTED)
    filed_date: datetime = Field(default_factory=datetime.now)
    # Ye field sirf internal use ke liye — kabhi features mein nahi jaata
    # Ground truth fraud_ring_id alag file mein store hota hai
    image_path: Optional[str] = Field(None,
                                       description="Claim image ka path, agar available hai")


class FraudRingGroundTruth(BaseModel):
    """
    Ground truth — SIRF evaluation ke liye, KABHI training features mein nahi.
    Ye alag CSV mein store hota hai — leakage prevention ka core mechanism.

    Interview answer: "The ground truth is stored in a separate file that is
    never joined into the feature matrix. The training code physically cannot
    access ring membership because it's not in the same dataframe."
    """
    claim_id: str = Field(..., pattern=r"^CLM-\d{6}$")
    fraud_ring_id: str = Field(..., pattern=r"^RING-\d{3}$")
    ring_type: str = Field(..., description="shared_device / shared_shop / shared_payment / mixed")
    is_fraud: bool = Field(default=True)


# === API Request/Response Schemas ===

class ClaimAnalyzeRequest(BaseModel):
    """API request: analyze a single claim"""
    claim_id: str = Field(..., pattern=r"^CLM-\d{6}$")


class RiskScoreResponse(BaseModel):
    """API response: risk assessment for a claim"""
    claim_id: str
    risk_level: RiskLevel
    risk_score: float = Field(..., ge=0.0, le=1.0)
    recommendation: InvestigationAction
    ml_probability: float
    graph_risk_indicator: bool
    multimodal_inconsistency: bool
    anomaly_score: float
    top_risk_factors: list[str]
    shap_values: Optional[dict] = None


class ChatRequest(BaseModel):
    """API request: chat with the investigator assistant"""
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """API response: assistant's reply"""
    response: str
    sources: list[str] = Field(default_factory=list)
    intent_detected: str
    session_id: str
    audit_trail: list[dict] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """API health check response"""
    status: str
    postgres_connected: bool
    neo4j_connected: bool
    model_loaded: bool
    version: str
