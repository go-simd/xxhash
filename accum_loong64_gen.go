//go:build ignore

// Command gen produces accum_loong64.s with go-asmgen: the XXH3 stripe
// accumulator on LSX (accumStripeLSX / accumRunLSX / accumScrambleLSX).
//
// LSX registers are 128-bit = two uint64 lanes, so a 64-byte stripe is four V
// registers (the accumulator pairs V1..V4). For each chunk:
//
//	dk  = dv ^ key
//	lo  = dk & 0xFFFFFFFF        (dk << 32 >> 32)
//	hi  = dk >> 32
//	acc += lo*hi                 (VMULV low 64 = full product, lo,hi < 2^32)
//	acc += swap-pair(dv)         (VSHUF4IW $0x4e swaps the two doublewords:
//	                             [w0,w1,w2,w3] -> [w2,w3,w0,w1])
//
// All integer add/mul, lane-neutral, so the digest matches every architecture.
//
// Three shapes are emitted, mirroring the amd64 kernel:
//
//   - accumStripeLSX(acc, p, sec): fold a single stripe (the final overlapping
//     tail stripe).
//   - accumRunLSX(acc, p, sec, nStripes): fold nStripes *consecutive* stripes in
//     one call, keeping the four accumulator pairs resident in V1..V4 across the
//     whole run (loaded once, stored once) and advancing the secret by 8 bytes
//     per stripe (the secret[s*8:] schedule). The accumulator pairs are no
//     longer round-tripped through memory per chunk, so the mul/add chains
//     software-pipeline instead of stalling on a per-stripe load/store and
//     Go/asm boundary — the lever that closes the ~0.67x loong64 gap.
//   - accumScrambleLSX(acc, p, sec, nBlocks): fold nBlocks complete 1024-byte
//     blocks (16 stripes + the inter-block scramble each) in one call, with the
//     accumulator resident across the WHOLE run and the scramble run in-register
//     (v ^= v>>47; v ^= secret[128+i*8]; v *= prime32_1 — a single VMULV, LSX
//     has a native 64-bit vector multiply). prime32_1 is broadcast to both lanes
//     via a small read-only constant table (primeWide).
//
// Run: go run accum_loong64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/loong64"
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

var accReg = []string{"V1", "V2", "V3", "V4"}

// stripeBody folds the four chunks of one stripe into the resident accumulator
// pairs accReg, reading dv from R5 (offset 16*c) and the secret from R6 (offset
// 16*c). Scratch: V5 (dv), V6 (key/dk), V7 (lo / mul), V8 (hi), V9 (swap).
// Clobbers R7.
func stripeBody(b *loong64.Builder) {
	for c := 0; c < 4; c++ {
		off := c * 16
		acc := accReg[c]
		b.
			Raw("MOVV $%d, R7", off).
			Raw("VMOVQ (R5)(R7), V5"). // dv
			Raw("VMOVQ (R6)(R7), V6"). // key
			Raw("VXORV V5, V6, V6").   // dk = dv ^ key
			Raw("VSLLV $32, V6, V7").
			Raw("VSRLV $32, V7, V7").      // lo = dk & 0xFFFFFFFF
			Raw("VSRLV $32, V6, V8").      // hi = dk >> 32
			Raw("VMULV V7, V8, V7").       // lo*hi
			Raw("VSHUF4IW $0x4e, V5, V9"). // swap the two u64 of dv
			Raw("VADDV " + acc + ", V7, " + acc).
			Raw("VADDV " + acc + ", V9, " + acc)
	}
}

// loadAcc / storeAcc move the four accumulator pairs between memory (R4) and the
// resident V1..V4. R7 is the index register.
func loadAcc(b *loong64.Builder) {
	for c := 0; c < 4; c++ {
		b.Raw("MOVV $%d, R7", c*16).Raw("VMOVQ (R4)(R7), %s", accReg[c])
	}
}
func storeAcc(b *loong64.Builder) {
	for c := 0; c < 4; c++ {
		b.Raw("MOVV $%d, R7", c*16).Raw("VMOVQ %s, (R4)(R7)", accReg[c])
	}
}

func main() {
	f := emit.NewFile("loong64")

	// prime32_1 broadcast across two 64-bit lanes, for the scramble's VMULV.
	primeBytes := []byte{
		0xb1, 0x79, 0x37, 0x9e, 0x00, 0x00, 0x00, 0x00,
		0xb1, 0x79, 0x37, 0x9e, 0x00, 0x00, 0x00, 0x00,
	}
	primeSym := f.Data("primeWide", primeBytes)

	// --- single stripe ---
	b := loong64.NewFunc("accumStripeLSX", stripeSig(), 0)
	b.LoadArg("acc", "R4").LoadArg("p_base", "R5").LoadArg("sec_base", "R6")
	loadAcc(b)
	stripeBody(b)
	storeAcc(b)
	b.Ret()
	f.Add(b.Func())

	// --- multi-stripe run: acc resident in V1..V4 across nStripes stripes. ---
	// R5 = p, R6 = sec, R8 = stripe counter. Each iteration advances p by 64 and
	// sec by 8 (the secret[s*8:] schedule).
	vr := loong64.NewFunc("accumRunLSX", runSig(), 0)
	vr.LoadArg("acc", "R4").LoadArg("p_base", "R5").LoadArg("sec_base", "R6").
		LoadArg("nStripes", "R8")
	loadAcc(vr)
	vr.Label("accumRunLSX_loop")
	stripeBody(vr)
	vr.
		Raw("ADDV $64, R5").
		Raw("ADDV $8, R6").
		Raw("ADDV $-1, R8").
		Raw("BNE R8, accumRunLSX_loop")
	storeAcc(vr)
	vr.Ret()
	f.Add(vr.Func())

	// --- full-block run: nBlocks complete 1024-byte blocks (16 stripes +
	// scramble) in one call, acc resident in V1..V4 across the WHOLE run. R5 = p
	// (advances continuously, 64/stripe = 1024/block), R9 = secret base (fixed),
	// R6 = working secret cursor (reset to R9 at the top of each block), R8 =
	// block counter. V11 = prime32_1 broadcast for the scramble multiply. ---
	vb := loong64.NewFunc("accumScrambleLSX", blockSig(), 0)
	vb.LoadArg("acc", "R4").LoadArg("p_base", "R5").LoadArg("sec_base", "R9").
		LoadArg("nBlocks", "R8")
	loadAcc(vb)
	vb.
		Raw("MOVV $%s+0(SB), R10", primeSym).
		Raw("VMOVQ (R10), V11"). // prime32_1 in both lanes
		Label("accumScrambleLSX_loop").
		Raw("MOVV R9, R6") // reset secret cursor to the block base
	for s := 0; s < 16; s++ {
		stripeBody(vb)
		vb.Raw("ADDV $64, R5").Raw("ADDV $8, R6")
	}
	// In-register scramble of the four pairs:
	//   v ^= v>>47; v ^= secret[128+c*16]; v *= prime32_1.
	for c := 0; c < 4; c++ {
		acc := accReg[c]
		scrOff := 128 + c*16
		vb.
			Raw("VSRLV $47, "+acc+", V5"). // v >> 47
			Raw("VXORV "+acc+", V5, V5").  // v ^= v>>47
			Raw("MOVV $%d, R7", scrOff).
			Raw("VMOVQ (R9)(R7), V6"). // secret chunk
			Raw("VXORV V5, V6, V5").   // v ^= secret
			Raw("VMULV V5, V11, " + acc)
	}
	// R5 already advanced by 16*64 = 1024 across the stripe loop (next block).
	vb.
		Raw("ADDV $-1, R8").
		Raw("BNE R8, accumScrambleLSX_loop")
	storeAcc(vb)
	vb.Ret()
	f.Add(vb.Func())

	if err := os.WriteFile("accum_loong64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote accum_loong64.s")
}
