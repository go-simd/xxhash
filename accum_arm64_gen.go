//go:build ignore

// Command gen produces accum_arm64.s with go-asmgen: the XXH3 stripe
// accumulator on NEON (accumStripeNEON / accumRunNEON / accumScrambleNEON).
//
// One stripe is 64 bytes = eight uint64 lanes, held as four V registers of two
// lanes each (V1..V4). For each 2-lane chunk: dk = dv ^ key; the 32x32->64
// product lo32(dk)*hi32(dk) is added to the lane, and the input word dv (with
// the two lanes of the pair exchanged via VEXT $8 — the acc[i^1] swap) is added
// too. XTN / SHRN / UMULL have no Go assembler mnemonics, so they are emitted as
// the documented WORD encodings (same approach as the upstream zeebo/xxh3 NEON
// kernel). The math is integer add/mul, lane-neutral, so the digest matches
// every other architecture.
//
// Three shapes are emitted, mirroring the amd64 kernel:
//
//   - accumStripeNEON(acc, p, sec): fold a single stripe (the final overlapping
//     tail stripe).
//   - accumRunNEON(acc, p, sec, nStripes): fold nStripes *consecutive* stripes
//     in one call, keeping the four accumulator banks V1..V4 resident in vector
//     registers across the whole run (loaded once, stored once) and advancing
//     the secret by 8 bytes per stripe (the secret[s*8:] schedule). The four
//     bank chains then software-pipeline against each other instead of stalling
//     on a per-stripe load/store and Go/asm boundary.
//   - accumScrambleNEON(acc, p, sec, nBlocks): fold nBlocks complete 1024-byte
//     blocks (16 stripes + the inter-block scramble each) in one call, with the
//     accumulator resident across the WHOLE multi-block run and the scramble run
//     in-register, so the long-input path crosses the Go/asm boundary once.
//     The scramble is v ^= v>>47; v ^= secret[128+i*8]; v *= prime32_1. NEON has
//     no 64-bit vector multiply, so v *= prime32_1 is split lo32*prime +
//     (hi32*prime)<<32 (prime32_1 < 2^32), the same full 64x32->64 as amd64.
//
// Run: go run accum_arm64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
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

// WORD encodings for the NEON ops the Go assembler lacks mnemonics for.
func xtn(d, n int) string  { return fmt.Sprintf("WORD $%#x", 0x0EA12800|(n<<5)|d) } // XTN  Vd.2S, Vn.2D -> low 32
func shrn(d, n int) string { return fmt.Sprintf("WORD $%#x", 0x0F208400|(n<<5)|d) } // SHRN #32, Vn.2D, Vd.2S -> high 32
func umull(d, n, m int) string {
	return fmt.Sprintf("WORD $%#x", 0x2EA0C000|(m<<16)|(n<<5)|d) // UMULL Vd.2D, Vn.2S, Vm.2S -> 32x32->64
}
func shl32(d, n int) string { // SHL Vd.2D, Vn.2D, #32  (left shift each doubleword by 32)
	return fmt.Sprintf("WORD $%#x", 0x4F605400|(n<<5)|d)
}

// stripeBody emits the four-chunk stripe accumulate over accumulator banks
// accReg, reading dv from base register pReg at dOff and the secret from sReg at
// sOff. Scratch: V5..V9.
func stripeBody(b *arm64.Builder, pReg, sReg string, accReg []string) {
	for c := 0; c < 4; c++ {
		off := c * 16
		acc := accReg[c]
		b.
			Raw("ADD $%d, %s, R5", off, pReg).
			Raw("VLD1 (R5), [V5.D2]"). // dv
			Raw("ADD $%d, %s, R5", off, sReg).
			Raw("VLD1 (R5), [V6.D2]").              // key
			Raw("VEOR V5.B16, V6.B16, V6.B16").     // dk = dv ^ key
			Raw(xtn(7, 6)).                         // V7 = lo32(dk)
			Raw(shrn(8, 6)).                        // V8 = hi32(dk)
			Raw(umull(9, 7, 8)).                    // V9 = lo32*hi32
			Raw("VEXT $8, V5.B16, V5.B16, V5.B16"). // swap the pair of dv
			Raw("VADD V5.D2, " + acc + ".D2, " + acc + ".D2").
			Raw("VADD V9.D2, " + acc + ".D2, " + acc + ".D2")
	}
}

func main() {
	accReg := []string{"V1", "V2", "V3", "V4"}

	// --- single stripe ---
	b := arm64.NewFunc("accumStripeNEON", stripeSig(), 0)
	b.LoadArg("acc", "R0").LoadArg("p_base", "R1").LoadArg("sec_base", "R2").
		Raw("VLD1 (R0), [V1.D2, V2.D2, V3.D2, V4.D2]")
	stripeBody(b, "R1", "R2", accReg)
	b.Raw("VST1 [V1.D2, V2.D2, V3.D2, V4.D2], (R0)").Ret()

	// --- multi-stripe run: acc resident in V1..V4 across nStripes stripes. ---
	// R1 = p, R2 = sec, R3 = stripe counter. Each iteration advances p by 64 and
	// sec by 8 (the secret[s*8:] schedule).
	vr := arm64.NewFunc("accumRunNEON", runSig(), 0)
	vr.LoadArg("acc", "R0").LoadArg("p_base", "R1").LoadArg("sec_base", "R2").
		LoadArg("nStripes", "R3").
		Raw("VLD1 (R0), [V1.D2, V2.D2, V3.D2, V4.D2]").
		Label("accumRunNEON_loop")
	stripeBody(vr, "R1", "R2", accReg)
	vr.
		Raw("ADD $64, R1, R1").
		Raw("ADD $8, R2, R2").
		Raw("SUB $1, R3, R3").
		Raw("CBNZ R3, accumRunNEON_loop").
		Raw("VST1 [V1.D2, V2.D2, V3.D2, V4.D2], (R0)").
		Ret()

	// --- full-block run: nBlocks complete 1024-byte blocks (16 stripes +
	// scramble) in one call, acc resident across the WHOLE run. R1 = p, R2 = sec
	// base (fixed across blocks), R3 = block counter. V0 = prime32_1 broadcast
	// for the scramble multiply. ---
	vb := arm64.NewFunc("accumScrambleNEON", blockSig(), 0)
	vb.LoadArg("acc", "R0").LoadArg("p_base", "R1").LoadArg("sec_base", "R2").
		LoadArg("nBlocks", "R3").
		Raw("VLD1 (R0), [V1.D2, V2.D2, V3.D2, V4.D2]").
		// prime32_1 broadcast into all four 32-bit lanes of V0 (UMULL reads the
		// low two .2S words of each operand, so the prime must sit in every S lane,
		// not just the low word of each doubleword).
		Raw("MOVD $0x9E3779B1, R6").
		Raw("VDUP R6, V0.S4").
		Label("accumScrambleNEON_loop")
	for s := 0; s < 16; s++ {
		dOff := s * 64
		sOff := s * 8
		for c := 0; c < 4; c++ {
			off := c * 16
			acc := accReg[c]
			vb.
				Raw("ADD $%d, R1, R5", dOff+off).
				Raw("VLD1 (R5), [V5.D2]").
				Raw("ADD $%d, R2, R5", sOff+off).
				Raw("VLD1 (R5), [V6.D2]").
				Raw("VEOR V5.B16, V6.B16, V6.B16").
				Raw(xtn(7, 6)).
				Raw(shrn(8, 6)).
				Raw(umull(9, 7, 8)).
				Raw("VEXT $8, V5.B16, V5.B16, V5.B16").
				Raw("VADD V5.D2, " + acc + ".D2, " + acc + ".D2").
				Raw("VADD V9.D2, " + acc + ".D2, " + acc + ".D2")
		}
	}
	// In-register scramble of the four banks:
	//   v ^= v>>47; v ^= secret[128+i*16 (chunk)]; v *= prime32_1.
	for c := 0; c < 4; c++ {
		acc := accReg[c]
		scrOff := 128 + c*16
		vb.
			Raw("VUSHR $47, "+acc+".D2, V5.D2").     // v >> 47
			Raw("VEOR V5.B16, "+acc+".B16, V5.B16"). // v ^= v>>47
			Raw("ADD $%d, R2, R5", scrOff).
			Raw("VLD1 (R5), [V6.D2]").          // secret chunk
			Raw("VEOR V6.B16, V5.B16, V5.B16"). // v ^= secret
			// v *= prime32_1 (full 64x32->64): lo32(v)*prime + (hi32(v)*prime)<<32.
			Raw(xtn(7, 5)).                          // V7 = lo32(v)
			Raw(shrn(8, 5)).                         // V8 = hi32(v)
			Raw(umull(6, 7, 0)).                     // V6 = lo32(v)*prime
			Raw(umull(9, 8, 0)).                     // V9 = hi32(v)*prime
			Raw(shl32(9, 9)).                        // V9 <<= 32
			Raw("VADD V9.D2, V6.D2, " + acc + ".D2") // v = lo*prime + (hi*prime)<<32
	}
	vb.
		Raw("ADD $1024, R1, R1"). // next block (secret base fixed)
		Raw("SUB $1, R3, R3").
		Raw("CBNZ R3, accumScrambleNEON_loop").
		Raw("VST1 [V1.D2, V2.D2, V3.D2, V4.D2], (R0)").
		Ret()

	f := emit.NewFile("arm64")
	f.Add(b.Func())
	f.Add(vr.Func())
	f.Add(vb.Func())
	if err := os.WriteFile("accum_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote accum_arm64.s")
}
