from pathlib import Path


def load_classes(path: str | Path) -> list[str]:
    return Path(path).read_text().split()


def label_maps(classes: list[str]) -> tuple[dict[int, str], dict[str, int]]:
    if len(set(classes)) != len(classes):
        raise ValueError(f"duplicate class names: {classes}")
    return dict(enumerate(classes)), {name: index for index, name in enumerate(classes)}


def category_to_label(categories: list[dict], label2id: dict[str, int]) -> dict[int, int]:
    """coco category id to model label id, matched by name; every category must be known"""
    unknown = [category["name"] for category in categories if category["name"] not in label2id]
    if unknown:
        raise ValueError(f"categories missing from the class list: {unknown}")
    return {category["id"]: label2id[category["name"]] for category in categories}
