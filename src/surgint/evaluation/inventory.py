from collections import Counter
from collections.abc import Iterable, Mapping, Sequence


def inventory_counts(
    frames: Iterable[tuple[Sequence[str], Sequence[int]]],
    predictions: Mapping[int, int],
    class_ids: Sequence[int],
) -> dict:
    """Primitive inventory counts for one complete session."""
    classes = _classes(class_ids)
    known = set(classes)
    instances: dict[str, int] = {}

    for instance_uids, frame_classes in frames:
        if len(instance_uids) != len(frame_classes):
            raise ValueError(
                "every ground-truth class needs an instance_uid: "
                f"{len(frame_classes)} classes with {len(instance_uids)} instance_uids"
            )
        for instance_uid, class_id in zip(instance_uids, frame_classes):
            if not isinstance(instance_uid, str) or not instance_uid.strip():
                raise ValueError("instance_uid values must be non-empty strings")
            class_id = int(class_id)
            if class_id not in known:
                raise ValueError(f"ground-truth class_id {class_id} is not in {classes}")
            previous = instances.setdefault(instance_uid, class_id)
            if previous != class_id:
                raise ValueError(
                    f"instance_uid {instance_uid!r} maps to classes {previous} and {class_id}"
                )

    gt_observed = Counter(instances.values())
    predicted = _predictions(predictions, known)
    ground_truth = {class_id: gt_observed.get(class_id, 0) for class_id in classes}
    predicted = {class_id: predicted.get(class_id, 0) for class_id in classes}
    errors = {class_id: predicted[class_id] - ground_truth[class_id] for class_id in classes}
    absolute_errors = {class_id: abs(errors[class_id]) for class_id in classes}
    exact = {class_id: int(errors[class_id] == 0) for class_id in classes}

    return {
        "classes": classes,
        "ground_truth": ground_truth,
        "predictions": predicted,
        "errors": errors,
        "absolute_errors": absolute_errors,
        "exact": exact,
        "session_exact": int(all(exact.values())),
    }


def inventory_evaluate(counts) -> dict:
    """Pool inventory count errors over one or more complete sessions."""
    if isinstance(counts, dict):
        counts = [counts]
    counts = list(counts)
    if not counts:
        raise ValueError("inventory evaluation requires at least one session")

    classes = tuple(counts[0]["classes"])
    if any(tuple(entry["classes"]) != classes for entry in counts[1:]):
        raise ValueError("inventory sessions must use the same class ids")

    per_class = {}
    for class_id in classes:
        ground_truth = sum(entry["ground_truth"][class_id] for entry in counts)
        predictions = sum(entry["predictions"][class_id] for entry in counts)
        absolute_error = sum(entry["absolute_errors"][class_id] for entry in counts)
        exact_matches = sum(entry["exact"][class_id] for entry in counts)
        per_class[class_id] = {
            "ground_truth": ground_truth,
            "predictions": predictions,
            "error": predictions - ground_truth,
            "absolute_error": absolute_error,
            "exact_matches": exact_matches,
            "exact_match_rate": exact_matches / len(counts),
        }

    comparisons = len(classes) * len(counts)
    exact_matches = sum(item["exact_matches"] for item in per_class.values())
    ground_truth = sum(item["ground_truth"] for item in per_class.values())
    predictions = sum(item["predictions"] for item in per_class.values())
    absolute_error = sum(item["absolute_error"] for item in per_class.values())

    return {
        "inventory_ground_truth": ground_truth,
        "inventory_predictions": predictions,
        "inventory_error": predictions - ground_truth,
        "inventory_absolute_error": absolute_error,
        "inventory_mean_absolute_error": absolute_error / comparisons if comparisons else 0.0,
        "class_exact_match_rate": exact_matches / comparisons if comparisons else 0.0,
        "session_exact_match_rate": sum(entry["session_exact"] for entry in counts) / len(counts),
        "inventory_per_class": per_class,
    }


def _classes(class_ids: Sequence[int]) -> tuple[int, ...]:
    classes = tuple(int(class_id) for class_id in class_ids)
    if len(set(classes)) != len(classes):
        raise ValueError(f"class ids must be unique, got {classes}")
    if any(class_id < 0 for class_id in classes):
        raise ValueError(f"class ids must be non-negative, got {classes}")
    return classes


def _predictions(predictions: Mapping[int, int], known: set[int]) -> dict[int, int]:
    if not isinstance(predictions, Mapping):
        raise TypeError("predictions must map class ids to counts")

    result = {}
    for class_id, count in predictions.items():
        class_id, count = int(class_id), int(count)
        if class_id not in known:
            raise ValueError(f"predicted class_id {class_id} is not in {sorted(known)}")
        if count < 0:
            raise ValueError(f"predicted count must be non-negative, got {count}")
        result[class_id] = count
    return result
