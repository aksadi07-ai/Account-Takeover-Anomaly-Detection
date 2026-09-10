def test_attack_types_are_defined():
    allowed = {
        "normal", "new_device", "new_location", "credential_attack",
        "beneficiary_attack", "velocity_attack", "large_transfer", "composite_ato"
    }
    assert "composite_ato" in allowed
    assert "normal" in allowed
