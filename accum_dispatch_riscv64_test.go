//go:build riscv64

package xxhash

import "testing"

// TestDispatchRISCV64 exercises both the RVV and the scalar fallback kernels.
func TestDispatchRISCV64(t *testing.T) {
	b := make([]byte, 1100)
	for i := range b {
		b[i] = byte(i)
	}
	want := xxh3Ref(b)
	saved := hasRVV
	defer func() { hasRVV = saved }()
	for _, v := range []bool{true, false} {
		hasRVV = v
		if got := Sum64(b); got != want {
			t.Fatalf("hasRVV=%v: Sum64 = %016x, want %016x", v, got, want)
		}
	}
}
