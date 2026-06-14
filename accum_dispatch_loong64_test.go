//go:build loong64

package xxhash

import "testing"

// TestDispatchLoong64 exercises both the LSX and the scalar fallback kernels.
func TestDispatchLoong64(t *testing.T) {
	b := make([]byte, 1100)
	for i := range b {
		b[i] = byte(i)
	}
	want := xxh3Ref(b)
	saved := hasLSX
	defer func() { hasLSX = saved }()
	for _, v := range []bool{true, false} {
		hasLSX = v
		if got := Sum64(b); got != want {
			t.Fatalf("hasLSX=%v: Sum64 = %016x, want %016x", v, got, want)
		}
	}
}
