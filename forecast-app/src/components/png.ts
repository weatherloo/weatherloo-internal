/**
 * Minimal PNG encoder (8-bit RGBA, uncompressed deflate blocks) returning a data URI.
 * Small rasters only; the map is 80 x 80. Scaling the image up gives smooth interpolation for free.
 */
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(bytes: Uint8Array, start: number, end: number) {
  let c = 0xffffffff;
  for (let i = start; i < end; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

const B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
function base64(bytes: Uint8Array) {
  let out = "";
  for (let i = 0; i < bytes.length; i += 3) {
    const n = (bytes[i] << 16) | ((bytes[i + 1] ?? 0) << 8) | (bytes[i + 2] ?? 0);
    out += B64[(n >> 18) & 63] + B64[(n >> 12) & 63];
    out += i + 1 < bytes.length ? B64[(n >> 6) & 63] : "=";
    out += i + 2 < bytes.length ? B64[n & 63] : "=";
  }
  return out;
}

export function pngDataUri(width: number, height: number, rgba: Uint8Array): string {
  // Raw scanlines, each prefixed with filter type 0.
  const rowLen = width * 4 + 1;
  const raw = new Uint8Array(rowLen * height);
  for (let y = 0; y < height; y++) raw.set(rgba.subarray(y * width * 4, (y + 1) * width * 4), y * rowLen + 1);

  // zlib stream of stored (uncompressed) deflate blocks, max 65535 bytes each.
  const blocks = Math.ceil(raw.length / 65535) || 1;
  const zlib = new Uint8Array(2 + raw.length + blocks * 5 + 4);
  let z = 0;
  zlib[z++] = 0x78; zlib[z++] = 0x01;
  for (let b = 0; b < blocks; b++) {
    const start = b * 65535, len = Math.min(65535, raw.length - start);
    zlib[z++] = b === blocks - 1 ? 1 : 0;
    zlib[z++] = len & 255; zlib[z++] = len >> 8;
    zlib[z++] = ~len & 255; zlib[z++] = (~len >> 8) & 255;
    zlib.set(raw.subarray(start, start + len), z);
    z += len;
  }
  let a = 1, b2 = 0;
  for (let i = 0; i < raw.length; i++) { a = (a + raw[i]) % 65521; b2 = (b2 + a) % 65521; }
  const adler = ((b2 << 16) | a) >>> 0;
  zlib[z++] = adler >>> 24; zlib[z++] = (adler >>> 16) & 255; zlib[z++] = (adler >>> 8) & 255; zlib[z++] = adler & 255;

  const chunk = (type: string, data: Uint8Array) => {
    const out = new Uint8Array(12 + data.length);
    const view = new DataView(out.buffer);
    view.setUint32(0, data.length);
    for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
    out.set(data, 8);
    view.setUint32(8 + data.length, crc32(out, 4, 8 + data.length));
    return out;
  };
  const header = new Uint8Array(13);
  const hv = new DataView(header.buffer);
  hv.setUint32(0, width); hv.setUint32(4, height);
  header[8] = 8; header[9] = 6; // 8-bit, RGBA

  const parts = [
    new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk("IHDR", header),
    chunk("IDAT", zlib.subarray(0, z)),
    chunk("IEND", new Uint8Array(0)),
  ];
  const png = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) { png.set(p, o); o += p.length; }
  return `data:image/png;base64,${base64(png)}`;
}
