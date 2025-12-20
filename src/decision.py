def recommend(pred_price, current_price, threshold=0.015):
    """BUY if predicted up > 1.5%, SELL if down > 1.5%, else HOLD"""
    diff = (pred_price - current_price) / current_price
    if diff > threshold:
        return "BUY", float(diff)
    elif diff < -threshold:
        return "SELL", float(diff)
    else:
        return "HOLD", float(diff)
