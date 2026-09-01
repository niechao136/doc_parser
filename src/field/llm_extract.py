import json
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from typing import List

from src.models.llm import llm
from src.utils.path import OUT_DIR


FIELD_DIR = OUT_DIR / "field"
FIELD_PATH = FIELD_DIR / "fields.json"
WORD_PATH = FIELD_DIR / "rec_box.json"
LLM_PATH = FIELD_DIR / "llm.json"


class PositionSchema(BaseModel):
    x: float
    y: float
    width: float | None
    height: float

class FieldSchema(BaseModel):
    fieldKey: str
    position: PositionSchema | None
    option: dict[str, PositionSchema] | None
    overwritePosition: PositionSchema | None



class OutputSchema(BaseModel):
    fields: List[FieldSchema]


format_llm = llm.with_structured_output(OutputSchema)


SYSTEM_PROMPT = """
你是一个表单解析助手，你会接收到一个表单栏位信息和表单图片的解析结果，图片的解析结果中会保存每个文字的坐标信息；
你要根据这些信息，解析出每个表单栏位的所需的坐标信息。
- 非选择类栏位（Text / Number / Date ...）：记录 x, y, height
    y / height 直接取该栏位标题所在 OCR 行的 box
    x 通常取标题冒号右边界再留出 10px 间隙；特殊内嵌横线栏位取横线起点

- 如果表单栏位已有默认值，position 仍是填写起点，另记录 overwritePosition
    作为清除默认文字的区域

- 选择类栏位（SingleChoice / MultiChoice）：为每个选项记录 x, y, width, height
    优先使用选项文字前面紧邻的 □ / ○ 符号的坐标
    如果 OCR 没有识别出符号（漏检/看漏），则用选项第一个字的框，
    按字符高度估算一个等大的方框位置，往左推算出来
"""



def main():
    system_message = SystemMessage(content=SYSTEM_PROMPT)
    fields_data = json.loads(FIELD_PATH.read_text(encoding="utf-8"))
    ocr_data = json.loads(WORD_PATH.read_text(encoding="utf-8"))
    input = f"""
表单信息：{fields_data}
坐标信息：{ocr_data}
"""
    human_message = HumanMessage(content=input)
    try:
        raw = format_llm.invoke([system_message, human_message])
    except Exception as e:
        print(f"Error invoking LLM: {e}")
        return
    if isinstance(raw, OutputSchema):
        result = raw
    else:
        result = OutputSchema.model_validate(raw)

    output = json.dumps(result.model_dump(), ensure_ascii=False, indent=4)
    LLM_PATH.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
