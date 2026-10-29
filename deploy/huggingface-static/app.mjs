import {SIZE, PIXELS, inputTensor, decodeClasses, unpackMask, scoreMask, paint} from './inference.mjs';

const byId = id => document.getElementById(id);
let examples = [], index = 0, session, cached = null, sequence = 0;
function controls(loading) {
  for (const id of ['scene', 'instruction']) byId(id).disabled = loading;
  byId('previous').disabled = loading || index === 0;
  byId('next').disabled = loading || index === examples.length - 1;
  byId('panels').setAttribute('aria-busy', String(loading));
}
function draw(id, pixels) {
  const canvas = document.createElement('canvas'); canvas.width = canvas.height = SIZE;
  canvas.getContext('2d').putImageData(new ImageData(pixels, SIZE, SIZE), 0, 0);
  byId(id).src = canvas.toDataURL('image/png');
}
async function imagePixels(source) {
  const image = new Image(); image.src = source; await image.decode();
  if (image.naturalWidth !== SIZE || image.naturalHeight !== SIZE) throw new Error('Unexpected example size.');
  const canvas = document.createElement('canvas'); canvas.width = canvas.height = SIZE;
  const context = canvas.getContext('2d', {willReadFrequently: true}); context.drawImage(image, 0, 0);
  return context.getImageData(0, 0, SIZE, SIZE).data;
}
async function show() {
  const request = ++sequence, row = examples[index], task = byId('instruction').value;
  controls(true); byId('status').textContent = 'Running the editor in your browser…';
  for (const id of ['match', 'edit', 'preserve', 'errors']) byId(id).textContent = '—';
  try {
    if (!cached || cached.id !== row.id) {
      const rgba = await imagePixels(row.input);
      const tensor = new ort.Tensor('float32', inputTensor(rgba), [1, 3, SIZE, SIZE]);
      const started = performance.now();
      const output = await session.run({images: tensor});
      const milliseconds = performance.now() - started;
      cached = {id: row.id, rgba, classes: decodeClasses(output.logits.data), milliseconds};
    }
    if (request !== sequence) return;
    // Only RGB enters session.run; these labels supply the target and scores.
    const requestedClass = task === 'circle' ? 1 : 2;
    const other = task === 'circle' ? 'square' : 'circle';
    const target = unpackMask(row[task + '_mask']), preserve = unpackMask(row[other + '_mask']);
    const result = scoreMask(cached.classes, requestedClass, target, preserve);
    draw('input', cached.rgba);
    draw('prediction', paint(cached.rgba, Uint8Array.from(cached.classes, c => c === requestedClass ? 1 : 0)));
    draw('target', paint(cached.rgba, target));
    byId('match').textContent = result.exact ? 'Exact match' : 'Contains errors';
    byId('match').className = 'score-value ' + (result.exact ? 'good' : 'imperfect');
    byId('edit').textContent = (result.editAccuracy * 100).toFixed(2) + '%';
    byId('preserve').textContent = (result.preservation * 100).toFixed(2) + '%';
    byId('errors').textContent = result.wrongPixels.toLocaleString();
    byId('edit-label').textContent = (task === 'circle' ? 'Circle' : 'Square') + ' edited correctly';
    byId('preserve-label').textContent = (other === 'circle' ? 'Circle' : 'Square') + ' preserved';
    byId('position').textContent = `${index + 1} of ${examples.length}`;
    byId('status').textContent = `${row.id} · Recolour ${task} · ${cached.milliseconds.toFixed(1)} ms browser inference`;
  } catch (error) { byId('status').textContent = 'Could not display this request: ' + error.message; }
  finally { if (request === sequence) controls(false); }
}
byId('scene').addEventListener('change', () => { index = byId('scene').selectedIndex; show(); });
byId('instruction').addEventListener('change', show);
for (const [id, step] of [['previous', -1], ['next', 1]]) byId(id).addEventListener('click', () => {
  index = Math.max(0, Math.min(examples.length - 1, index + step)); byId('scene').selectedIndex = index; show();
});

try {
  const get = async path => { const r = await fetch(path); if (!r.ok) throw new Error(`Download failed: ${path}`); return r; };
  const metadata = await (await get('./model_metadata.json')).json();
  const model = await (await get('./model.onnx')).arrayBuffer();
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', model)), b => b.toString(16).padStart(2, '0')).join('');
  if (digest !== metadata.onnx_sha256) throw new Error('The model checksum does not match the frozen export.');
  ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = new URL('./vendor/', document.baseURI).href;
  session = await ort.InferenceSession.create(new Uint8Array(model), {executionProviders: ['wasm']});
  examples = await (await get('./examples.json')).json();
  if (!examples.length) throw new Error('No validation scenes available.');
  for (const row of examples) byId('scene').add(new Option(row.id, row.id));
  await show();
} catch (error) { byId('status').textContent = 'Could not load the demo: ' + error.message; }
