"""美妆护肤场景的安全边界与风险提示。"""

from dataclasses import dataclass


_URGENT_TERMS = (
    "呼吸困难",
    "喉咙肿",
    "喉头肿",
    "眼睛肿",
    "大面积水疱",
    "流脓",
    "出血不止",
    "迅速扩散",
    "严重疼痛",
    "高热",
    "急诊",
)

_MEDICAL_TERMS = (
    "孕期",
    "孕妇",
    "备孕",
    "哺乳",
    "儿童",
    "婴儿",
    "湿疹",
    "银屑病",
    "玫瑰痤疮",
    "荨麻疹",
    "过敏",
    "处方",
    "药物",
    "激素",
    "维a酸",
    "异维a酸",
    "抗生素",
)


@dataclass(frozen=True)
class SafetyAssessment:
    """对当前问题的最小化、可解释风险判断。"""

    flags: tuple[str, ...] = ()
    notice: str = ""
    instruction: str = (
        "只提供一般性的护肤和成分信息，不进行疾病诊断，不替代医生；"
        "不要编造功效、处方或药物剂量。对严重、持续或不确定的症状，建议咨询专业人士。"
    )
    requires_clinician: bool = False


def assess_question(question: str | None) -> SafetyAssessment:
    """基于明确的风险词给模型和用户提供保守边界提示。

    这不是医学分类器，也不尝试推断疾病；它只负责在高风险语境出现时
    提醒模型收窄回答范围，并把用户引导到专业帮助。
    """

    normalized = str(question or "").strip().lower()
    flags: list[str] = []
    if any(term in normalized for term in _URGENT_TERMS):
        flags.append("urgent")
    if any(term in normalized for term in _MEDICAL_TERMS):
        flags.append("medical")

    if "urgent" in flags:
        return SafetyAssessment(
            flags=tuple(flags),
            notice=(
                "安全提醒：你描述的情况可能需要及时就医。Beauty Agent 不能诊断或处理急症，"
                "请尽快联系皮肤科或急诊，不要仅依据本回答自行用药。"
            ),
            instruction=(
                "这是可能需要及时处理的健康问题。不要诊断、推荐处方或延误就医；"
                "先建议用户联系皮肤科或急诊，只能补充非常一般的护理注意事项。"
            ),
            requires_clinician=True,
        )

    if "medical" in flags:
        return SafetyAssessment(
            flags=tuple(flags),
            notice=(
                "安全提醒：问题涉及孕期、儿童、过敏、用药或疑似皮肤疾病。"
                "以下内容仅是一般护肤信息，不是诊断；使用活性成分前请先咨询医生或药师。"
            ),
            instruction=(
                "问题涉及特殊人群、药物或疑似皮肤疾病。不要做诊断或给出药物剂量，"
                "明确建议用户先咨询医生或药师，并提醒进行局部测试和停止刺激性产品。"
            ),
            requires_clinician=True,
        )

    return SafetyAssessment()

