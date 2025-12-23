import re

def normalize_name(name: str) -> str:
    """
    Standardize drug names (remove suffixes like _box, _rot90)
    """
    name = name.lower()
    name = re.sub(r'_box.*', '', name)
    name = re.sub(r'_blister.*', '', name)
    name = re.sub(r'_pack.*', '', name)
    name = re.sub(r'_rot.*', '', name)
    name = re.sub(r'[^a-z0-9]', '', name)
    return name