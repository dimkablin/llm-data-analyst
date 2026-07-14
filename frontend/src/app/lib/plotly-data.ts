type PlotlyTypedArrayPayload = {
  dtype?: unknown;
  bdata?: unknown;
};

function decodeBase64Bytes(value: string): Uint8Array {
  const binary = atob(value);
  return Uint8Array.from(binary, (char) => char.charCodeAt(0));
}

export function plotlySequence(value: unknown): unknown[] {
  if (Array.isArray(value)) {
    return value;
  }
  if (!value || typeof value !== "object") {
    return [];
  }

  const payload = value as PlotlyTypedArrayPayload;
  if (typeof payload.dtype !== "string" || typeof payload.bdata !== "string") {
    return [];
  }

  const bytes = decodeBase64Bytes(payload.bdata);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const dtype = payload.dtype.replace(/[<>=|]/g, "").toLowerCase();
  const out: number[] = [];

  const read = (size: number, getter: (offset: number) => number | bigint) => {
    for (let offset = 0; offset + size <= view.byteLength; offset += size) {
      out.push(Number(getter(offset)));
    }
  };

  if (dtype === "f8" || dtype === "float64") read(8, (offset) => view.getFloat64(offset, true));
  else if (dtype === "f4" || dtype === "float32") read(4, (offset) => view.getFloat32(offset, true));
  else if (dtype === "i4" || dtype === "int32") read(4, (offset) => view.getInt32(offset, true));
  else if (dtype === "u4" || dtype === "uint32") read(4, (offset) => view.getUint32(offset, true));
  else if (dtype === "i2" || dtype === "int16") read(2, (offset) => view.getInt16(offset, true));
  else if (dtype === "u2" || dtype === "uint16") read(2, (offset) => view.getUint16(offset, true));
  else if (dtype === "i1" || dtype === "int8") read(1, (offset) => view.getInt8(offset));
  else if (dtype === "u1" || dtype === "uint8") read(1, (offset) => view.getUint8(offset));
  else if (dtype === "i8" || dtype === "int64") read(8, (offset) => view.getBigInt64(offset, true));
  else if (dtype === "u8" || dtype === "uint64") read(8, (offset) => view.getBigUint64(offset, true));

  return out;
}
