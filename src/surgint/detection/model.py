from transformers import RTDetrForObjectDetection, RTDetrImageProcessor


def build_model(checkpoint: str, id2label: dict[int, str] | None = None) -> RTDetrForObjectDetection:
    """load rt-detr; pass id2label to replace the pretrained classification head"""
    if id2label is None:
        return RTDetrForObjectDetection.from_pretrained(checkpoint)

    return RTDetrForObjectDetection.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id={name: index for index, name in id2label.items()},
        ignore_mismatched_sizes=True,
    )


def build_processor(checkpoint: str) -> RTDetrImageProcessor:
    return RTDetrImageProcessor.from_pretrained(checkpoint)
