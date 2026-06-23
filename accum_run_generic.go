//go:build !amd64

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
