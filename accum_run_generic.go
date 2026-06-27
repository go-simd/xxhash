//go:build !amd64 && !arm64 && !riscv64

package xxhash

// accumRun folds nStripes consecutive stripes starting at p, with the secret
// advancing by 8 bytes per stripe (the secret[s*8:] schedule). On architectures
// other than amd64 it loops the per-stripe kernel; the amd64 build provides a
// register-resident multi-stripe kernel instead (accum_amd64.go). nStripes must
// be >= 1.
func accumRun(acc *[8]uint64, p, sec []byte, nStripes int) {
	for s := 0; s < nStripes; s++ {
		accumStripe(acc, p[s*stripeBytes:], sec[s*8:])
	}
}

// accumScramble is the portable form of the amd64 in-asm full-block kernel: it
// folds nBlocks complete 1024-byte blocks (16 stripes + scramble each) using the
// per-stripe SIMD kernel and the Go-side scramble. The amd64 build replaces this
// with a single asm call that keeps the accumulator resident across the whole
// run and runs the scramble in-register (accum_amd64.go). nBlocks must be >= 1.
func accumScramble(acc *[8]uint64, p, sec []byte, nBlocks int) {
	for i := 0; i < nBlocks; i++ {
		blk := p[i*blockBytes:]
		accumRun(acc, blk, sec, blockStripes)
		scramble(acc)
	}
}
