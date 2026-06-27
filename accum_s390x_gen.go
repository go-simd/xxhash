//go:build ignore

// Command gen produces accum_s390x.s with go-asmgen: the XXH3 stripe
// accumulator on the z/Architecture vector facility (accumStripeVEC /
// accumRunVEC / accumScrambleVEC). The vector facility is baseline on z13+, so
// there is no runtime dispatch.
//
// s390x is the only BIG-ENDIAN target. XXH3 reads its input little-endian, but
// VL loads bytes in memory order into big-endian elements, so a freshly loaded
// doubleword holds the byte-reversed value. We therefore byte-reverse each
// 8-byte lane right after loading (VPERM with a per-doubleword reversal
// selector) so the vector lanes carry the true little-endian values — exactly
// what the Go scalar path obtains via binary.LittleEndian. From there the math
// is identical to every other architecture and lane-neutral.
//
// Per 16-byte chunk (two uint64 lanes):
//
//	dk  = dv ^ key
//	hi  = dk >> 32                 (VESRLG $32, per doubleword)
//	lo  = (dk << 32) >> 32         (low 32 bits of dk)
//	acc += lo*hi                   (VMLOF: odd-word unsigned widening multiply
//	                               32x32->64; the low 32 bits of each doubleword
//	                               are the odd fullword in big-endian element
//	                               order, lo,hi < 2^32 so it is exact)
//	acc += swap-lanes(dv)          (VPDI $4 exchanges the two doublewords = the
//	                               acc[i^1] += dv mapping)
//
// The accumulator is a Go [8]uint64; VST writes lane VALUES in the host's native
// big-endian layout, which is how Go reads the uint64s back, so only the INPUT
// (and the scramble's secret chunk) need the byte reversal.
//
// Three shapes are emitted, mirroring the amd64 kernel:
//
//   - accumStripeVEC(acc, p, sec): fold a single stripe (the final overlapping
//     tail stripe).
//   - accumRunVEC(acc, p, sec, nStripes): fold nStripes *consecutive* stripes in
//     one call, keeping the four accumulator pairs resident in V24..V27 across
//     the whole run (loaded once, stored once) and advancing the secret by 8
//     bytes per stripe (the secret[s*8:] schedule). The accumulator pairs are no
//     longer round-tripped through memory per chunk.
//   - accumScrambleVEC(acc, p, sec, nBlocks): fold nBlocks complete 1024-byte
//     blocks (16 stripes + the inter-block scramble each) in one call, with the
//     accumulator resident across the WHOLE run and the scramble run in-register
//     (v ^= v>>47; v ^= secret[128+i*8]; v *= prime32_1 — full 64x32->64 via two
//     VMLOF on the odd words). The scramble secret chunk is byte-reversed like
//     the input so its lane values are little-endian.
//
// Bit-exactness (this is the big-endian risk) is the gate pinned by the qemu
// official-vector and differential tests.
//
// Run: go run accum_s390x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
)

func stripeSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Scalar("acc", abi.Int64), abi.Slice("p"), abi.Slice("sec")},
		[]abi.Arg{},
	)
}

func runSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{
			abi.Scalar("acc", abi.Int64), abi.Slice("p"), abi.Slice("sec"),
			abi.Scalar("nStripes", abi.Int64),
		},
		[]abi.Arg{},
	)
}

func blockSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{
			abi.Scalar("acc", abi.Int64), abi.Slice("p"), abi.Slice("sec"),
			abi.Scalar("nBlocks", abi.Int64),
		},
		[]abi.Arg{},
	)
}

var accReg = []string{"V24", "V25", "V26", "V27"}

// stripeBody folds the four chunks of one stripe into the resident accumulator
// pairs accReg, reading dv from R2 (offset 16*c) and the secret from R3 (offset
// 16*c). V31 holds the byte-reversal selector. Scratch: V0 (dv), V1 (key),
// V2 (dk), V3, V4, V5, V6.
func stripeBody(b *s390x.Builder) {
	for c := 0; c < 4; c++ {
		off := c * 16
		acc := accReg[c]
		b.
			Raw("VL %d(R2), V0", off).    // dv (raw, big-endian elements)
			Raw("VL %d(R3), V1", off).    // key
			Raw("VPERM V0, V0, V31, V0"). // byte-reverse each lane -> true LE values
			Raw("VPERM V1, V1, V31, V1").
			Raw("VX V0, V1, V2").       // dk = dv ^ key
			Raw("VESRLG $32, V2, V4").  // hi = dk >> 32
			Raw("VESLG $32, V2, V3").   // dk << 32
			Raw("VESRLG $32, V3, V3").  // lo = low 32 of dk
			Raw("VMLOF V3, V4, V5").    // lo*hi per doubleword (odd-word widening)
			Raw("VPDI $4, V0, V0, V6"). // swap the two lanes of dv
			Raw("VAG " + acc + ", V5, " + acc).
			Raw("VAG " + acc + ", V6, " + acc)
	}
}

func loadAcc(b *s390x.Builder) {
	for c := 0; c < 4; c++ {
		b.Raw("VL %d(R1), %s", c*16, accReg[c])
	}
}
func storeAcc(b *s390x.Builder) {
	for c := 0; c < 4; c++ {
		b.Raw("VST %s, %d(R1)", accReg[c], c*16)
	}
}

func main() {
	f := emit.NewFile("s390x")

	// --- single stripe ---
	b := s390x.NewFunc("accumStripeVEC", stripeSig(), 0)
	b.LoadArg("acc", "R1").LoadArg("p_base", "R2").LoadArg("sec_base", "R3").
		Raw("MOVD $revsel<>(SB), R5").
		Raw("VL (R5), V31")
	loadAcc(b)
	stripeBody(b)
	storeAcc(b)
	b.Ret()
	f.Add(b.Func())

	// --- multi-stripe run: acc resident in V24..V27 across nStripes stripes. ---
	// R2 = p, R3 = sec, R4 = stripe counter. Each iteration advances p by 64 and
	// sec by 8 (the secret[s*8:] schedule).
	vr := s390x.NewFunc("accumRunVEC", runSig(), 0)
	vr.LoadArg("acc", "R1").LoadArg("p_base", "R2").LoadArg("sec_base", "R3").
		LoadArg("nStripes", "R4").
		Raw("MOVD $revsel<>(SB), R5").
		Raw("VL (R5), V31")
	loadAcc(vr)
	vr.Label("accumRunVEC_loop")
	stripeBody(vr)
	vr.
		Raw("ADD $64, R2").
		Raw("ADD $8, R3").
		Raw("ADD $-1, R4").
		Raw("CMPBNE R4, $0, accumRunVEC_loop")
	storeAcc(vr)
	vr.Ret()
	f.Add(vr.Func())

	// --- full-block run: nBlocks complete 1024-byte blocks (16 stripes +
	// scramble) in one call, acc resident in V24..V27 across the WHOLE run. R2 =
	// p (advances continuously, 64/stripe = 1024/block), R6 = secret base
	// (fixed), R3 = working secret cursor (reset to R6 at the top of each block),
	// R4 = block counter. V30 = prime32_1 broadcast in the odd word of each
	// doubleword for the scramble multiply; V31 = byte-reversal selector. ---
	vb := s390x.NewFunc("accumScrambleVEC", blockSig(), 0)
	vb.LoadArg("acc", "R1").LoadArg("p_base", "R2").LoadArg("sec_base", "R6").
		LoadArg("nBlocks", "R4").
		Raw("MOVD $revsel<>(SB), R5").
		Raw("VL (R5), V31").
		Raw("MOVD $primeWide<>(SB), R5").
		Raw("VL (R5), V30")
	loadAcc(vb)
	vb.Label("accumScrambleVEC_loop").
		Raw("MOVD R6, R3") // reset secret cursor to the block base
	for s := 0; s < 16; s++ {
		stripeBody(vb)
		vb.Raw("ADD $64, R2").Raw("ADD $8, R3")
	}
	// In-register scramble of the four pairs:
	//   v ^= v>>47; v ^= secret[128+c*16]; v *= prime32_1.
	// The scramble secret chunk is loaded from the fixed base R6 and byte-reversed
	// (V31) so its lane values are little-endian, matching the scalar path.
	for c := 0; c < 4; c++ {
		acc := accReg[c]
		scrOff := 128 + c*16
		vb.
			Raw("VESRLG $47, "+acc+", V2"). // v >> 47
			Raw("VX "+acc+", V2, V2").      // v ^= v>>47
			Raw("VL %d(R6), V3", scrOff).   // secret chunk (raw, big-endian)
			Raw("VPERM V3, V3, V31, V3").   // -> true LE lane values
			Raw("VX V2, V3, V2").           // v ^= secret
			// v *= prime32_1 (full 64x32->64): lo32(v) and hi32(v) each in the odd
			// word; lo*prime (VMLOF) + (hi*prime << 32) (VMLOF then VESLG by 32).
			Raw("VESLG $32, V2, V4").  // v << 32
			Raw("VESRLG $32, V4, V4"). // lo32(v) (odd word)
			Raw("VESRLG $32, V2, V5"). // hi32(v) (odd word)
			Raw("VMLOF V4, V30, V6").  // lo32(v)*prime
			Raw("VMLOF V5, V30, V5").  // hi32(v)*prime
			Raw("VESLG $32, V5, V5").  // (hi32(v)*prime) << 32
			Raw("VAG V6, V5, " + acc)  // v = lo*prime + (hi*prime)<<32
	}
	// R2 already advanced by 16*64 = 1024 across the stripe loop (next block).
	vb.Raw("ADD $-1, R4").
		Raw("CMPBNE R4, $0, accumScrambleVEC_loop")
	storeAcc(vb)
	vb.Ret()
	f.Add(vb.Func())

	out := f.String()
	// Append the reversal selector constant used by VPERM.
	out += "\nGLOBL revsel<>(SB), RODATA|NOPTR, $16\n"
	for i := 0; i < 16; i++ {
		// lane-local byte reversal: position i takes source byte (i&^7)|(7-(i&7)).
		src := (i &^ 7) | (7 - (i & 7))
		out += fmt.Sprintf("DATA revsel<>+%d(SB)/1, $%d\n", i, src)
	}
	// prime32_1 in the odd (low) 32-bit word of each big-endian doubleword: bytes
	// 4..7 and 12..15 hold 0x9E3779B1 big-endian; the high words are zero.
	out += "\nGLOBL primeWide<>(SB), RODATA|NOPTR, $16\n"
	primeBE := []byte{0x9e, 0x37, 0x79, 0xb1}
	for i := 0; i < 16; i++ {
		var v byte
		switch {
		case i >= 4 && i < 8:
			v = primeBE[i-4]
		case i >= 12 && i < 16:
			v = primeBE[i-12]
		}
		out += fmt.Sprintf("DATA primeWide<>+%d(SB)/1, $%d\n", i, v)
	}
	if err := os.WriteFile("accum_s390x.s", []byte(out), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote accum_s390x.s")
}
