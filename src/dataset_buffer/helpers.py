"""Out of the box helper functions."""


def label_1_only(row: dict) -> bool:
    """Inspects a dict

    Args:
        row (dict): the row to inspect

    Returns:
        bool: True if label == 1
    """
    label = row.get("label", 0)
    return label == 1


def label_0_only(row: dict) -> bool:
    """Inspects a dict

    Args:
        row (dict): the row to inspect

    Returns:
        bool: True if label == 1
    """
    label = row.get("label", 1)
    return label == 0
