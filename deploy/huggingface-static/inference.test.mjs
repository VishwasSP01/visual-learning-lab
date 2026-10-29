import test from 'node:test';
import assert from 'node:assert/strict';
import {PIXELS, inputTensor, decodeClasses, unpackMask, scoreMask, paint} from './inference.mjs';

test('NCHW input uses RGB pixels only', () => {
  const rgba = new Uint8ClampedArray(PIXELS * 4); rgba.set([255, 0, 128, 255]);
  const result = inputTensor(rgba);
  assert.equal(result[0], 1); assert.equal(result[PIXELS], 0);
  assert.equal(result[2 * PIXELS], Math.fround(128 / 255));
});
test('argmax uses lowest-index ties and rejects non-finite scores', () => {
  const values = new Float32Array(PIXELS * 3);
  assert.equal(decodeClasses(values)[0], 0);
  values[PIXELS] = values[2 * PIXELS] = 2; assert.equal(decodeClasses(values)[0], 1);
  values[2 * PIXELS] = 3; assert.equal(decodeClasses(values)[0], 2);
  values[4] = NaN; assert.throws(() => decodeClasses(values), /Non-finite/);
});
test('packed mask bit order, preservation, wrong pixels, and rendering', () => {
  const bytes = new Uint8Array(PIXELS / 8); bytes[0] = 128;
  const target = unpackMask(btoa(String.fromCharCode(...bytes)));
  assert.equal(target[0], 1); assert.equal(target[1], 0);
  const preserve = new Uint8Array(PIXELS); preserve[1] = 1;
  const classes = new Uint8Array(PIXELS); classes[0] = 1; classes[1] = 2;
  assert.deepEqual(scoreMask(classes, 1, target, preserve), {wrongPixels: 0, exact: true, editAccuracy: 1, preservation: 1});
  classes[1] = 1; assert.equal(scoreMask(classes, 1, target, preserve).wrongPixels, 1);
  assert.equal(scoreMask(classes, 1, target, preserve).preservation, 0);
  const rgba = new Uint8ClampedArray(PIXELS * 4); rgba.fill(255);
  const painted = paint(rgba, target);
  assert.deepEqual(Array.from(painted.slice(0, 4)), [255, 0, 0, 255]);
  assert.deepEqual(Array.from(painted.slice(4, 8)), [255, 255, 255, 255]);
});
