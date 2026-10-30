# gen-core-image

> Package: `sceneworks-gen-core-image` · library: `gen_core_image`

Tensor-free **HDR image I/O**: OpenEXR read/write (`exr_io`) and the HDR conversions (`hdr`: the
HLG master, 10-bit YUV packing, the VAE-range maps and the EXR payload conversions). The colour
policy a request opts into — `HdrColorSpace`, its primaries and working-space transfer — stays in
`gen_core::hdr` and is re-exported here.

Split out of [`gen-core`](../gen-core/README.md) in sc-27316. It depends on gen-core (gen-core
never depends on it) and, like gen-core, on no tensor library. With it goes `exr`, which leaves the
iOS graph.

## License

Apache-2.0.
