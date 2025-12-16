# defining how conversation data is structured and validated

from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class Turn(BaseModel):
    turn_id: int = Field(..., description="Turn index(0-indexed)")
    speaker: str = Field(...,
                         description="Speaker role (agent/customer/system)")  # e.g., 'agent', 'customer', 'system'
    # text of the utterance
    text: str = Field(..., description="Utterance text")
    timestamp: Optional[float] = Field(
        None, description="Timestamp in seconds")  # optional timestamp of the utterance
    start_time: Optional[float] = Field(
        None, description="Start time in seconds")  # optional start time of the utterance
    # optional end time of the utterance
    end_time: Optional[float] = Field(None, description="End time in seconds")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata")  # optional metadata for extensibility

    # Dialog act annotations for multiwoz dataset
    dialog_acts: Optional[Dict[str, List[List[str]]]
                          ] = Field(None, description="Dialog acts")  # e.g., {'Inform': [['restaurant', 'area', 'north']]}
    span_info: Optional[List[List[Any]]] = Field(
        None, description="Span annotations")  # e.g., [[start_idx, end_idx, 'slot_name']]
    frames: Optional[List[Dict[str, Any]]] = Field(
        None, description="MultiWOZ frames (state tracking)")  # e.g., [{'service': 'hotel', 'state': {...}, 'actions': [...]}]
    slots: Optional[List[Dict[str, Any]]] = Field(
        None, description="Slot annotations")  # e.g., [{'slot': 'price', 'value': 'cheap'}]


# event detection labels
class EventLabel(BaseModel):
    event_type: str = Field(...,
                            description="Event type (escalation/refund/churn/etc)")  # e.g., 'escalation', 'refund', 'churn'
    confidence: float = Field(
        1.0, description="Confidence score", ge=0.0, le=1.0)  # confidence score between 0 and 1
    span_turn_ids: List[int] = Field(
        default_factory=list, description="Turns where event is evident")  # list of turn IDs where the event is evident
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata")  # optional metadata for extensibility


# evidence span annotations
class EvidenceSpanAnnotation(BaseModel):
    # e.g., turn index
    turn_id: int = Field(..., description="Turn containing the evidence")
    # e.g., start character index of the evidence span
    char_start: int = Field(..., description="Character start position")
    # e.g., end character index of the evidence span
    char_end: int = Field(..., description="Character end position")
    # e.g., text of the evidence span
    text: str = Field(..., description="Evidence text")
    # e.g., 'escalation', 'refund', 'churn'
    event_type: str = Field(..., description="Related event type")
    confidence: float = Field(
        1.0, description="Annotation confidence", ge=0.0, le=1.0)  # confidence score between 0 and 1

# normalizing conversation object


class Conversation(BaseModel):
    conversation_id: str = Field(..., description="Unique conversation ID")
    turns: List[Turn] = Field(..., description="List of conversation turns")

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Conversation metadata")
    date: Optional[str] = Field(None, description="Conversation date")
    channel: Optional[str] = Field(
        None, description="Channel (phone/chat/email)")
    agent_id: Optional[str] = Field(None, description="Agent identifier")
    customer_id: Optional[str] = Field(None, description="Customer identifier")
    duration_seconds: Optional[float] = Field(
        None, description="Total duration")

    # MultiWOZ specific
    domains: List[str] = Field(
        default_factory=list, description="Conversation domains")
    goal: Optional[Dict[str, Any]] = Field(
        None, description="Conversation goal")
    services: List[str] = Field(
        default_factory=list, description="Services discussed")

    # Event annotations (for training/evaluation)
    event_labels: List[EventLabel] = Field(
        default_factory=list, description="Event labels")
    evidence_spans: List[EvidenceSpanAnnotation] = Field(
        default_factory=list, description="Evidence spans")

    # Synthetic/derived fields
    is_synthetic: bool = Field(
        False, description="Whether conversation is synthetic")
    source_dataset: str = Field("multiwoz", description="Source dataset")
    synthetic_event_method: Optional[str] = Field(
        None,
        description="Method used to generate synthetic events (rule-based/keyword/llm)"
    )


# batch of conversations
class ConversationBatch(BaseModel):
    conversations: List[Conversation] = Field(
        ..., description="List of conversations")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Batch metadata")


# Annotation guidelines for labeling
class AnnotationGuideline(BaseModel):
    """Annotation guidelines for labeling"""
    event_type: str = Field(..., description="Event type to annotate")
    description: str = Field(..., description="Event description")
    examples: List[str] = Field(..., description="Example phrases")
    negative_examples: List[str] = Field(
        default_factory=list, description="Counter-examples")
    annotation_rules: List[str] = Field(...,
                                        description="Rules for annotation")


# event based guidelines
EVENT_GUIDELINES = {
    "escalation": AnnotationGuideline(
        event_type="escalation",
        description="Customer requests to speak with supervisor or escalates complaint",
        examples=[
            "I want to speak to your manager",
            "Transfer me to a supervisor",
            "This is unacceptable, let me talk to someone else",
            "I need to escalate this issue"
        ],
        negative_examples=[
            "Can you help me?",
            "I have a question"
        ],
        annotation_rules=[
            "Mark the turn where escalation is first requested",
            "Include turns with explicit supervisor/manager requests",
            "Include turns expressing extreme dissatisfaction leading to escalation"
        ]
    ),
    "refund": AnnotationGuideline(
        event_type="refund",
        description="Customer requests refund or money back",
        examples=[
            "I want a refund",
            "Can I get my money back?",
            "I'd like to return this for a refund",
            "Please refund my account"
        ],
        negative_examples=[
            "What is your refund policy?",
            "Do you offer refunds?"
        ],
        annotation_rules=[
            "Mark the turn where refund is explicitly requested",
            "Include both full and partial refund requests",
            "Distinguish from refund policy inquiries"
        ]
    ),
    "churn": AnnotationGuideline(
        event_type="churn",
        description="Customer indicates intent to cancel service or leave",
        examples=[
            "I want to cancel my subscription",
            "I'm switching to another provider",
            "Close my account please",
            "I'm done with this service"
        ],
        negative_examples=[
            "How do I cancel if I need to?",
            "What's your cancellation policy?"
        ],
        annotation_rules=[
            "Mark the turn where churn intent is expressed",
            "Include both explicit cancellation requests and implicit churn signals",
            "Distinguish from policy questions"
        ]
    ),
    "product_issue": AnnotationGuideline(
        event_type="product_issue",
        description="Customer reports problem with product or service",
        examples=[
            "The product is broken",
            "This doesn't work",
            "I'm having issues with the service",
            "The feature isn't working properly"
        ],
        negative_examples=[
            "How do I use this feature?",
            "What does this do?"
        ],
        annotation_rules=[
            "Mark the turn where issue is first reported",
            "Include defects, malfunctions, and quality issues",
            "Distinguish from usage questions"
        ]
    ),
    "billing_issue": AnnotationGuideline(
        event_type="billing_issue",
        description="Customer has question or complaint about billing/charges",
        examples=[
            "I was charged twice",
            "This bill is incorrect",
            "Why is my charge higher than expected?",
            "I didn't authorize this payment"
        ],
        negative_examples=[
            "When is my next bill?",
            "How much does this cost?"
        ],
        annotation_rules=[
            "Mark the turn where billing concern is raised",
            "Include incorrect charges, unexpected charges, and billing disputes",
            "Distinguish from general pricing inquiries"
        ]
    ),
    "booking_failure": AnnotationGuideline(
        event_type="booking_failure",
        description="Failed to complete booking or reservation",
        examples=[
            "There are no available slots",
            "I couldn't find anything matching your criteria",
            "The booking system is down",
            "All rooms are booked"
        ],
        annotation_rules=[
            "Mark when system cannot fulfill booking request",
            "Include both availability issues and system failures"
        ]
    ),

    "requirement_not_met": AnnotationGuideline(
        event_type="requirement_not_met",
        description="Service cannot meet customer requirements",
        examples=[
            "We don't have that amenity",
            "No restaurants in that price range",
            "No trains available at that time"
        ],
        annotation_rules=[
            "Mark when customer needs cannot be satisfied"
        ]
    )
}
