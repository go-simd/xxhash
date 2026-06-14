package xxhash

import (
	"bytes"
	"encoding/binary"
	"hash"
	"math/rand"
	"testing"

	"github.com/zeebo/xxh3"
)

// canonical anchors published by the upstream xxHash project.
func TestCanonicalAnchors(t *testing.T) {
	if got := Sum64(nil); got != 0x2d06800538d394c2 {
		t.Errorf(`Sum64("") = %016x, want 2d06800538d394c2`, got)
	}
	if got := Sum64([]byte("a")); got != 0xe6c632b61e964e1f {
		t.Errorf(`Sum64("a") = %016x, want e6c632b61e964e1f`, got)
	}
	if got := Sum64String(""); got != 0x2d06800538d394c2 {
		t.Errorf(`Sum64String("") = %016x`, got)
	}
}

// reference verifies bit-exactness against zeebo/xxh3 (documented to match the
// reference C implementation) across every length class, including the boundary
// lengths between hash paths and several long-input block boundaries.
func TestReference(t *testing.T) {
	r := rand.New(rand.NewSource(99))
	lens := []int{}
	for i := 0; i <= 260; i++ {
		lens = append(lens, i)
	}
	lens = append(lens, 511, 512, 513, 1023, 1024, 1025, 1087, 1088, 1089,
		2047, 2048, 2049, 4095, 4096, 4097, 9000, 65536, 200000)
	for _, n := range lens {
		b := make([]byte, n)
		r.Read(b)
		if got, want := Sum64(b), xxh3.Hash(b); got != want {
			t.Fatalf("len %d: Sum64 = %016x, want %016x", n, got, want)
		}
		if got, want := Sum64String(string(b)), xxh3.HashString(string(b)); got != want {
			t.Fatalf("len %d: Sum64String = %016x, want %016x", n, got, want)
		}
	}
}

// streaming checks the hash.Hash64 path against the one-shot result for many
// write-chunk patterns, including writes that straddle block boundaries.
func TestStreaming(t *testing.T) {
	r := rand.New(rand.NewSource(123))
	chunks := []int{1, 2, 7, 31, 63, 64, 65, 100, 1000, 1024, 1025, 4096}
	for _, n := range []int{0, 1, 16, 17, 64, 240, 241, 1024, 1025, 1088, 1089, 3000, 70000} {
		b := make([]byte, n)
		r.Read(b)
		want := Sum64(b)
		for _, cs := range chunks {
			d := New()
			for i := 0; i < len(b); i += cs {
				e := i + cs
				if e > len(b) {
					e = len(b)
				}
				if _, err := d.Write(b[i:e]); err != nil {
					t.Fatal(err)
				}
			}
			if got := d.Sum64(); got != want {
				t.Fatalf("len %d chunk %d: streaming = %016x, want %016x", n, cs, got, want)
			}
		}
		// WriteString must match Write.
		d := New()
		_, _ = d.WriteString(string(b))
		if got := d.Sum64(); got != want {
			t.Fatalf("len %d: WriteString = %016x, want %016x", n, got, want)
		}
	}
}

// interfaces and the Sum/Reset/Size/BlockSize surface.
func TestHashInterface(t *testing.T) {
	var h hash.Hash64 = New()
	_, _ = h.Write([]byte("hello world, this is a reasonably sized input"))
	if h.Size() != 8 {
		t.Errorf("Size = %d, want 8", h.Size())
	}
	if h.BlockSize() != 64 {
		t.Errorf("BlockSize = %d, want 64", h.BlockSize())
	}
	want := h.Sum64()
	var tmp [8]byte
	binary.BigEndian.PutUint64(tmp[:], want)
	if got := h.Sum(nil); !bytes.Equal(got, tmp[:]) {
		t.Errorf("Sum = %x, want %x", got, tmp[:])
	}
	// Sum must not change state.
	if h.Sum64() != want {
		t.Error("Sum64 changed after Sum")
	}
	// Reset returns to the empty digest.
	h.Reset()
	if h.Sum64() != Sum64(nil) {
		t.Error("Reset did not restore empty state")
	}
}

// FuzzSum64 is the differential gate: the SIMD/scalar Sum64 must equal the
// reference for arbitrary inputs on every architecture (this is what catches a
// big-endian or lane-order mistake under qemu).
func FuzzSum64(f *testing.F) {
	for _, s := range []string{"", "a", "abc", "the quick brown fox"} {
		f.Add([]byte(s))
	}
	f.Add(bytes.Repeat([]byte{0xA5}, 1025))
	f.Add(bytes.Repeat([]byte{0x5A}, 5000))
	f.Fuzz(func(t *testing.T, b []byte) {
		if got, want := Sum64(b), xxh3.Hash(b); got != want {
			t.Fatalf("len %d: Sum64 = %016x, want %016x", len(b), got, want)
		}
	})
}

// TestScalarStripe exercises accumStripeScalar directly. It is the active kernel
// on architectures without a SIMD path and the fallback when a CPU lacks the
// vector feature; testing it everywhere keeps coverage at 100% on every arch and
// proves the portable kernel folds a stripe identically to the long-input path
// for a one-block input.
func TestScalarStripe(t *testing.T) {
	p := make([]byte, stripeBytes)
	s := secret[:stripeBytes]
	for i := range p {
		p[i] = byte(i*31 + 7)
	}
	var got, want [8]uint64
	got = initAcc
	want = initAcc
	accumStripeScalar(&got, p, s)
	// Reference fold of one stripe, computed independently here.
	for i := 0; i < 8; i++ {
		dv := readU64(p[i*8:])
		dk := dv ^ readU64(s[i*8:])
		want[i^1] += dv
		want[i] += uint64(uint32(dk)) * (dk >> 32)
	}
	if got != want {
		t.Fatalf("accumStripeScalar = %v, want %v", got, want)
	}
}

// xxh3Ref computes the digest with the portable scalar stripe kernel, used by
// the per-architecture dispatch tests as an independent oracle for the SIMD
// path (it does not depend on which CPU feature flags are toggled).
func xxh3Ref(b []byte) uint64 {
	if len(b) <= 240 {
		return Sum64(b)
	}
	acc := initAcc
	nBlocks := (len(b) - 1) / blockBytes
	for i := 0; i < nBlocks; i++ {
		blk := b[i*blockBytes:]
		for s := 0; s < blockStripes; s++ {
			accumStripeScalar(&acc, blk[s*stripeBytes:], secret[s*8:])
		}
		scramble(&acc)
	}
	rest := b[nBlocks*blockBytes:]
	nStripes := (len(rest) - 1) / stripeBytes
	for s := 0; s < nStripes; s++ {
		accumStripeScalar(&acc, rest[s*stripeBytes:], secret[s*8:])
	}
	accumStripeScalar(&acc, b[len(b)-stripeBytes:], secret[secretBytes-stripeBytes-7:])
	out := uint64(len(b)) * prime64_1
	out += mulFold64(acc[0]^readU64(secret[11:]), acc[1]^readU64(secret[19:]))
	out += mulFold64(acc[2]^readU64(secret[27:]), acc[3]^readU64(secret[35:]))
	out += mulFold64(acc[4]^readU64(secret[43:]), acc[5]^readU64(secret[51:]))
	out += mulFold64(acc[6]^readU64(secret[59:]), acc[7]^readU64(secret[67:]))
	return xxh3Avalanche(out)
}
