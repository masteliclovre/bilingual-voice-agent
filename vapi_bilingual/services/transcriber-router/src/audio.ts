export function stereoToMono16le(stereo: Buffer): Buffer {
    // 2 bytes/sample, 2 channels interleaved: L R L R ...
    const len = stereo.length / 2; // in samples per channel? Actually total samples (16-bit words)
    const out = Buffer.alloc(len); // mono 16-bit -> same half bytes count? We'll compute properly
    const viewIn = new DataView(stereo.buffer, stereo.byteOffset, stereo.byteLength);
    const outView = new DataView(out.buffer, out.byteOffset, out.byteLength);
    let o = 0;
    for (let i = 0; i < stereo.length; i += 4) {
        const l = viewIn.getInt16(i, true);
        const r = viewIn.getInt16(i + 2, true);
        const m = (l + r) >> 1;
        outView.setInt16(o, m, true);
        o += 2;
    }
    return out.slice(0, o);
}