from transformers import RTDetrForObjectDetection


def load_model(checkpoint: str) -> RTDetrForObjectDetection:
    """load a checkpoint as it was saved; the head keeps whatever classes it has"""
    return RTDetrForObjectDetection.from_pretrained(checkpoint)


def load_model_with_new_head(checkpoint: str, id2label: dict[int, str]) -> RTDetrForObjectDetection:
    """load a checkpoint and resize every class head to id2label, reinitializing them"""
    return RTDetrForObjectDetection.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id={name: index for index, name in id2label.items()},
        ignore_mismatched_sizes=True,
    )
