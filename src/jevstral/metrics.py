"""Accuracy, Brier score and expected calibration error (ECE)."""

BINS = 10


def summarize(predictions: list[tuple[list[float], int]]) -> dict:
    """predictions: (option probabilities, index of the correct option) for each question."""
    if not predictions:
        return {"questions": 0}
    correct, brier, top = [], [], []
    for probabilities, label in predictions:
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        correct.append(best == label)
        top.append(probabilities[best])
        brier.append(sum((p - (i == label)) ** 2 for i, p in enumerate(probabilities)))
    return {
        "questions": len(predictions),
        "accuracy": sum(correct) / len(correct),
        "brier": sum(brier) / len(brier),
        "ece": expected_calibration_error(top, correct),
    }


def expected_calibration_error(confidences: list[float], correct: list[bool]) -> float:
    """Put the predictions in 10 equal bins by confidence.

    Return the mean |accuracy - confidence| of the bins, weighted by bin size.
    """
    bins: list[list[int]] = [[] for _ in range(BINS)]
    for i, confidence in enumerate(confidences):
        bins[min(int(confidence * BINS), BINS - 1)].append(i)
    error = 0.0
    for members in bins:
        if members:
            accuracy = sum(correct[i] for i in members) / len(members)
            confidence = sum(confidences[i] for i in members) / len(members)
            error += len(members) / len(confidences) * abs(accuracy - confidence)
    return error
