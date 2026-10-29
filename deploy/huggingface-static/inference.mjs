// Pure preprocessing, decoding, scoring, and rendering shared by app and tests.
export const SIZE = 64, PIXELS = SIZE * SIZE;

export function inputTensor(rgba) {
  if (rgba.length !== PIXELS * 4) throw new Error('Expected a 64 × 64 image.');
  const input = new Float32Array(3 * PIXELS);
  for (let i = 0; i < PIXELS; i++) for (let c = 0; c < 3; c++) input[c * PIXELS + i] = rgba[i * 4 + c] / 255;
  return input;
}

export function decodeClasses(logits) {
  if (logits.length !== 3 * PIXELS) throw new Error('Invalid model output size.');
  const classes = new Uint8Array(PIXELS);
  for (let i = 0; i < PIXELS; i++) {
    let best = 0;
    for (let c = 0; c < 3; c++) {
      const value = logits[c * PIXELS + i];
      if (!Number.isFinite(value)) throw new Error('Non-finite model output.');
      if (value > logits[best * PIXELS + i]) best = c;
    }
    classes[i] = best;
  }
  return classes;
}

export function unpackMask(base64) {
  const bytes = Uint8Array.from(atob(base64), c => c.charCodeAt(0));
  if (bytes.length !== PIXELS / 8) throw new Error('Invalid label-mask size.');
  return Uint8Array.from({length: PIXELS}, (_, i) => (bytes[i >> 3] >> (7 - (i & 7))) & 1);
}

export function scoreMask(classes, requestedClass, target, preserve) {
  let wrongPixels = 0, editPixels = 0, correctEdit = 0, oppositePixels = 0, unchangedOpposite = 0;
  for (let i = 0; i < PIXELS; i++) {
    const edited = classes[i] === requestedClass;
    if (edited !== Boolean(target[i])) wrongPixels++;
    if (target[i]) { editPixels++; if (edited) correctEdit++; }
    if (preserve[i]) { oppositePixels++; if (!edited) unchangedOpposite++; }
  }
  if (!editPixels || !oppositePixels) throw new Error('Expected both labelled objects.');
  return {wrongPixels, exact: wrongPixels === 0, editAccuracy: correctEdit / editPixels,
          preservation: unchangedOpposite / oppositePixels};
}

export function paint(rgba, mask) {
  const pixels = new Uint8ClampedArray(rgba);
  for (let i = 0; i < PIXELS; i++) if (mask[i]) {
    pixels[i * 4] = 255; pixels[i * 4 + 1] = 0; pixels[i * 4 + 2] = 0;
  }
  return pixels;
}
