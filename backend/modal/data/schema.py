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
