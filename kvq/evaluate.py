"""Answer extraction and scoring for the accuracy runs."""
import re


def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def normalize_number(s):
    if s is None:
        return None
    s = s.replace("\\!", "").replace(",", "").replace("$", "").replace("\\%", "").strip()
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"^\\?\(|\\?\)$", "", s).strip()
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return s
    x = float(m[-1])
    return str(int(x)) if x == int(x) else str(x)


def math_correct(text, answer):
    pred = last_boxed(text)
    if pred is None:
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        pred = nums[-1] if nums else None
    return normalize_number(pred) == normalize_number(answer)
