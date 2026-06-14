//go:build !amd64 && !arm64 && !riscv64 && !loong64 && !ppc64le && !s390x

package xxhash

// accumStripe is the portable scalar kernel on architectures without a SIMD
// implementation.
func accumStripe(acc *[8]uint64, p, sec []byte) {
	accumStripeScalar(acc, p, sec)
}
