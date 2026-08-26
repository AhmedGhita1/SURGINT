import pytest

from surgint.dataset.labels import category_to_label, label_maps, load_classes

CLASSES = ["scalpel", "scissors", "forceps"]


def test_label_maps_are_inverses():
    id2label, label2id = label_maps(CLASSES)

    print(f"{id2label}")
    assert id2label == {0: "scalpel", 1: "scissors", 2: "forceps"}
    assert label2id == {"scalpel": 0, "scissors": 1, "forceps": 2}


def test_duplicate_class_names_are_rejected():
    with pytest.raises(ValueError):
        label_maps(["scalpel", "scalpel"])


def test_categories_map_by_name_not_position():
    # coco ids run 1..3 but in a different order than the class list
    categories = [
        {"id": 1, "name": "forceps"},
        {"id": 2, "name": "scalpel"},
        {"id": 3, "name": "scissors"},
    ]
    _, label2id = label_maps(CLASSES)

    mapping = category_to_label(categories, label2id)

    print(f"category ids {[c['id'] for c in categories]}: labels {list(mapping.values())}")
    assert mapping == {1: 2, 2: 0, 3: 1}


def test_category_ids_need_not_be_contiguous():
    categories = [{"id": 7, "name": "scalpel"}, {"id": 41, "name": "forceps"}]
    _, label2id = label_maps(CLASSES)

    assert category_to_label(categories, label2id) == {7: 0, 41: 2}


def test_unknown_category_raises_instead_of_being_skipped():
    categories = [{"id": 1, "name": "scalpel"}, {"id": 2, "name": "drill"}]
    _, label2id = label_maps(CLASSES)

    with pytest.raises(ValueError, match="drill"):
        category_to_label(categories, label2id)


def test_load_classes_reads_one_name_per_line(tmp_path):
    path = tmp_path / "classes.txt"
    path.write_text("scalpel\nscissors\nforceps\n")

    assert load_classes(path) == CLASSES


def test_unit_labels():
    for test in [
        test_label_maps_are_inverses,
        test_duplicate_class_names_are_rejected,
        test_categories_map_by_name_not_position,
        test_category_ids_need_not_be_contiguous,
        test_unknown_category_raises_instead_of_being_skipped,
    ]:
        print(f"\n{test.__name__}")
        test()


if __name__ == "__main__":
    test_unit_labels()
    print("\nall passed")
