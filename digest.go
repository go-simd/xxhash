package xxhash

import (
	"encoding/binary"
	"hash"
)

// Digest is a streaming XXH3-64 hasher. The zero value is NOT ready for use;
// obtain one with New. Digest implements hash.Hash and hash.Hash64.
type Digest struct {
	acc    [8]uint64                      // running stripe accumulator
	buf    [blockBytes + stripeBytes]byte // staging for the current (partial) block
	buffed int                            // bytes currently in buf
	total  uint64                         // total bytes written
}

var (
	_ hash.Hash   = (*Digest)(nil)
	_ hash.Hash64 = (*Digest)(nil)
)

// New returns a new streaming XXH3-64 Digest.
func New() *Digest {
	d := new(Digest)
	d.Reset()
	return d
}

// Reset clears the Digest to its initial state.
func (d *Digest) Reset() {
	d.acc = initAcc
	d.buffed = 0
	d.total = 0
}

// Size returns the digest length in bytes (8).
func (d *Digest) Size() int { return 8 }

// BlockSize returns the hash's natural block size.
func (d *Digest) BlockSize() int { return stripeBytes }

// Write adds more data to the running hash. It never returns an error.
//
// Invariant: d.buf[:d.buffed] always holds the tail of the input not yet folded
// into a full 1024-byte block, and d.buffed stays in 1..blockBytes once any data
// has been written. We never consume the block that the buffer is exactly full
// with — we wait until strictly more arrives — so the final stripe is always the
// genuinely-last stripe of the input, matching the one-shot end-aligned read.
func (d *Digest) Write(p []byte) (int, error) {
	n := len(p)
	d.total += uint64(n)

	// Fast path: first writes, more than a full buffer pending and nothing
	// staged yet — fold whole blocks straight from p without copying.
	for d.buffed == 0 && len(p) > len(d.buf) {
		d.consumeBlock(p[:blockBytes])
		p = p[blockBytes:]
	}

	for len(p) > 0 {
		// Keep filling the staging buffer until it holds a full block plus a
		// trailing stripe; only then is it safe to fold one block, because the
		// retained trailing stripe guarantees the final stripe is never the one
		// folded into the accumulator early.
		if d.buffed < len(d.buf) {
			c := copy(d.buf[d.buffed:], p)
			d.buffed += c
			p = p[c:]
			continue
		}
		d.consumeBlock(d.buf[:blockBytes])
		// Carry the trailing stripe back to the front.
		d.buffed = copy(d.buf[:stripeBytes], d.buf[blockBytes:])
	}
	return n, nil
}

// WriteString adds more data to the running hash. It never returns an error.
func (d *Digest) WriteString(s string) (int, error) {
	return d.Write([]byte(s))
}

// consumeBlock accumulates one full 1024-byte block (16 stripes + scramble).
func (d *Digest) consumeBlock(blk []byte) {
	for s := 0; s < blockStripes; s++ {
		accumStripe(&d.acc, blk[s*stripeBytes:], secret[s*8:])
	}
	scramble(&d.acc)
}

// Sum64 returns the current 64-bit digest without altering the hash state.
func (d *Digest) Sum64() uint64 {
	// Short total: identical to the one-shot small-input paths.
	if d.total <= 240 {
		return Sum64(d.shortInput())
	}

	acc := d.acc
	tail := d.buf[:d.buffed]
	off := 0

	// The staging buffer may still hold a full 1024-byte block plus a trailing
	// remainder (Write only folds a block once the buffer overflows). Fold that
	// block here, with its scramble, before the end-aligned tail.
	if len(tail) > blockBytes {
		for s := 0; s < blockStripes; s++ {
			accumStripe(&acc, tail[s*stripeBytes:], secret[s*8:])
		}
		scramble(&acc)
		off = blockBytes
	}
	accumTail(&acc, tail, off)

	out := d.total * prime64_1
	out += mulFold64(acc[0]^readU64(secret[11:]), acc[1]^readU64(secret[19:]))
	out += mulFold64(acc[2]^readU64(secret[27:]), acc[3]^readU64(secret[35:]))
	out += mulFold64(acc[4]^readU64(secret[43:]), acc[5]^readU64(secret[51:]))
	out += mulFold64(acc[6]^readU64(secret[59:]), acc[7]^readU64(secret[67:]))
	return xxh3Avalanche(out)
}

// shortInput returns the full input when total <= 240, which fits entirely in
// the buffer (no block was ever consumed in that case).
func (d *Digest) shortInput() []byte {
	return d.buf[:d.total]
}

// Sum appends the big-endian digest to b. It does not alter the hash state.
func (d *Digest) Sum(b []byte) []byte {
	var tmp [8]byte
	binary.BigEndian.PutUint64(tmp[:], d.Sum64())
	return append(b, tmp[:]...)
}
