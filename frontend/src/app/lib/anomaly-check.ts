import type { AnomalyCheck } from "./backend-types";

export type CheckedTextPart = {
  text: string;
  item?: AnomalyCheck["items"][number];
};

export function splitCheckedNumbers(
  text: string,
  items: AnomalyCheck["items"],
): CheckedTextPart[] {
  const candidates = items.filter((item) => item.text.length > 0);
  const parts: CheckedTextPart[] = [];
  let cursor = 0;

  const findWholeNumber = (value: string) => {
    let index = text.indexOf(value, cursor);
    while (index >= 0) {
      const before = text[index - 1] ?? "";
      const after = text[index + value.length] ?? "";
      if (!/\d/.test(before) && !/\d/.test(after)) return index;
      index = text.indexOf(value, index + 1);
    }
    return -1;
  };

  while (cursor < text.length) {
    const next = candidates
      .map((item) => ({ item, index: findWholeNumber(item.text) }))
      .filter(({ index }) => index >= 0)
      .sort((left, right) => left.index - right.index || right.item.text.length - left.item.text.length)[0];

    if (!next) {
      parts.push({ text: text.slice(cursor) });
      break;
    }
    if (next.index > cursor) {
      parts.push({ text: text.slice(cursor, next.index) });
    }
    parts.push({ text: next.item.text, item: next.item });
    cursor = next.index + next.item.text.length;
  }

  return parts;
}
