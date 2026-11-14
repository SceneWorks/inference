import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const module = await WebAssembly.compile(readFileSync(process.argv[2]));
assert.equal(WebAssembly.Module.imports(module).length, 0, 'policy WASM must need no host imports');
const instance = await WebAssembly.instantiate(module, {});
assert.equal(instance.exports.run_corpus(), 21);
console.log('MOSS policy WASM corpus: 21 cases passed, zero imports');
