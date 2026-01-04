def multi_modal_decision(image=None, video=None, audio=None, sync=None):
    score = 0
    weight = 0

    if image:
        score += 0.2 * image["image_fake_probability"]
        weight += 0.2
    if video:
        score += 0.4 * video["video_fake_probability"]
        weight += 0.4
    if audio:
        score += 0.2 * audio["audio_fake_probability"]
        weight += 0.2
    if sync:
        score += 0.2 * (1 if sync["lip_sync_fake"] else 0)
        weight += 0.2

    final = score / weight

    decision = "BLOCK" if final > 0.65 else "FLAG" if final > 0.45 else "ALLOW"

    return {
        "final_risk_score": round(final, 3),
        "decision": decision
    }
