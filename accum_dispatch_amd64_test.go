//go:build amd64

package xxhash

import "testing"

// TestDispatchAMD64 exercises both the AVX2 and the SSE2 stripe kernels so that
// coverage of accumStripe is complete regardless of the host's AVX2 support.
func TestDispatchAMD64(t *testing.T) {
	b := make([]byte, 1100)
	for i := range b {
		b[i] = byte(i)
	}
	want := xxh3Ref(b)
	saved := hasAVX2
	defer func() { hasAVX2 = saved }()
	for _, avx2 := range []bool{true, false} {
		hasAVX2 = avx2
		if got := Sum64(b); got != want {
			t.Fatalf("hasAVX2=%v: Sum64 = %016x, want %016x", avx2, got, want)
		}
	}
}
