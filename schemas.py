from typing import Literal


PROMPT_TYPE = list[tuple[Literal["system"], str] | tuple[Literal["human"], str]]
