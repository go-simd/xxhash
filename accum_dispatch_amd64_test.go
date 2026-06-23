//go:build amd64

package xxhash

import "testing"

// TestDispatchAMD64 exercises both the AVX2 and the SSE2 kernels — the
// single-stripe, multi-stripe run, and full-block (stripes + scramble) shapes —
// so that coverage of accumStripe/accumRun/accumScramble is complete regardless
// of the host's AVX2 support. The lengths span multiple 1024-byte block
// boundaries plus partial-stripe and overlapping-tail cases, so each kernel is
// verified byte-exact against the reference across every block-loop path.
func TestDispatchAMD64(t *testing.T) {
	lens := []int{
		241, 256, 511, 512, 513, 1023, 1024, 1025, 1088, 1100,
		2048, 2049, 3072, 4096, 4097, 9000, 65536,
	}
	saved := hasAVX2
	defer func() { hasAVX2 = saved }()
	for _, n := range lens {
		b := make([]byte, n)
		for i := range b {
			b[i] = byte(i)
		}
		want := xxh3Ref(b)
		for _, avx2 := range []bool{true, false} {
			hasAVX2 = avx2
			if got := Sum64(b); got != want {
				t.Fatalf("len=%d hasAVX2=%v: Sum64 = %016x, want %016x", n, avx2, got, want)
			}
		}
	}
}
