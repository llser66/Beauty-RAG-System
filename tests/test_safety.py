from utils.safety import assess_question


def test_regular_beauty_question_has_no_medical_alert():
    assessment = assess_question("油皮适合什么质地的保湿霜？")

    assert assessment.flags == ()
    assert assessment.notice == ""


def test_pregnancy_question_gets_cautious_notice():
    assessment = assess_question("孕期可以使用视黄醇吗？")

    assert "medical" in assessment.flags
    assert assessment.requires_clinician is True
    assert "不是诊断" in assessment.notice
    assert "医生" in assessment.instruction


def test_urgent_symptoms_get_strong_referral():
    assessment = assess_question("脸部突然肿胀并且呼吸困难怎么办？")

    assert "urgent" in assessment.flags
    assert assessment.requires_clinician is True
    assert "及时就医" in assessment.notice

