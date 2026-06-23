//go:build ignore

// Command gen produces accum_amd64.s with go-asmgen: the XXH3 stripe
// accumulator, both AVX2 and SSE2.
//
// One stripe is 64 bytes = eight uint64 lanes. For each lane the kernel computes
// dk = dv ^ sec, adds the 32x32->64 product lo32(dk)*hi32(dk) to lane i, and
// adds the input word dv to lane i^1 (the adjacent-lane swap). At 128-bit
// granularity the swap is a within-pair u64 exchange (PSHUFD $0x4e); the product
// uses PSHUFD $0x31 to bring each u64's high half down, then [V]PMULUDQ.
//
// Two shapes are emitted per ISA:
//
//   - accumStripe{AVX2,SSE2}(acc, p, sec): fold a single stripe. Used for the
//     final overlapping tail stripe (which reads the last 64 bytes of the whole
//     input at a fixed secret offset).
//   - accumRun{AVX2,SSE2}(acc, p, sec, nStripes): fold nStripes *consecutive*
//     stripes in one call, keeping the eight accumulator lanes resident in
//     vector registers across the whole run and advancing the secret by 8 bytes
//     per stripe (the secret[s*8:] schedule). This is the multi-accumulator
//     shape: the four independent 256-bit (AVX2) / 128-bit (SSE2) accumulator
//     banks are never round-tripped through memory inside the loop, so the four
//     mul/add dependency chains software-pipeline against each other instead of
//     stalling on a per-stripe load/store. nStripes is assumed >= 1.
//
// The math is integer add/mul only, lane-neutral, so the digest is identical on
// every architecture. Input bytes are loaded as raw 64-bit words here; the
// little-endian decode that the spec mandates is already handled by the Go side
// for the short paths, and for the stripe path the lane values are combined only
// by endian-neutral add/xor/mul, so the final per-lane bytes feed back through
// the Go merge which reads them little-endian — bit-exactness is pinned by the
// official-vector and differential tests.
//
// Run: go run accum_amd64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/amd64"
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

func main() {
	f := emit.NewFile("amd64")

	// prime32_1 (0x9E3779B1) broadcast across four 64-bit lanes, used by the
	// in-asm block scramble's full 64x32->64 multiply. Declared once as a
	// file-local read-only constant table.
	primeBytes := []byte{
		0xb1, 0x79, 0x37, 0x9e, 0x00, 0x00, 0x00, 0x00,
		0xb1, 0x79, 0x37, 0x9e, 0x00, 0x00, 0x00, 0x00,
		0xb1, 0x79, 0x37, 0x9e, 0x00, 0x00, 0x00, 0x00,
		0xb1, 0x79, 0x37, 0x9e, 0x00, 0x00, 0x00, 0x00,
	}
	primeSym := f.Data("scramblePrime", primeBytes) // "scramblePrime<>"

	// --- AVX2 single stripe: process all 8 lanes as two YMM registers. ---
	v := amd64.NewFunc("accumStripeAVX2", stripeSig(), 0)
	v.LoadArg("acc", "AX").LoadArg("p_base", "CX").LoadArg("sec_base", "DX").
		Raw("VMOVDQU (AX), Y0").   // acc[0:4]
		Raw("VMOVDQU 32(AX), Y1"). // acc[4:8]
		// low half (lanes 0..3)
		Raw("VMOVDQU (CX), Y2").   // dv0
		Raw("VPXOR (DX), Y2, Y3"). // dk0 = dv0 ^ sec0
		Raw("VPSHUFD $0x31, Y3, Y4").
		Raw("VPMULUDQ Y3, Y4, Y3").   // dk0.lo * dk0.hi
		Raw("VPSHUFD $0x4e, Y2, Y2"). // swap64(dv0)
		Raw("VPADDQ Y0, Y3, Y0").
		Raw("VPADDQ Y0, Y2, Y0").
		// high half (lanes 4..7)
		Raw("VMOVDQU 32(CX), Y5").
		Raw("VPXOR 32(DX), Y5, Y6").
		Raw("VPSHUFD $0x31, Y6, Y7").
		Raw("VPMULUDQ Y6, Y7, Y6").
		Raw("VPSHUFD $0x4e, Y5, Y5").
		Raw("VPADDQ Y1, Y6, Y1").
		Raw("VPADDQ Y1, Y5, Y1").
		Raw("VMOVDQU Y0, (AX)").
		Raw("VMOVDQU Y1, 32(AX)").
		Raw("VZEROUPPER").
		Ret()
	f.Add(v.Func())

	// --- AVX2 multi-stripe run: acc resident in Y0/Y1 across nStripes stripes. ---
	// CX = p, DX = sec, BX = stripe counter. Each iteration advances p by 64 and
	// sec by 8 (the secret[s*8:] schedule). The two accumulator banks Y0 (lanes
	// 0..3) and Y1 (lanes 4..7) stay in registers for the whole loop.
	vr := amd64.NewFunc("accumRunAVX2", runSig(), 0)
	vr.LoadArg("acc", "AX").LoadArg("p_base", "CX").LoadArg("sec_base", "DX").
		LoadArg("nStripes", "BX").
		Raw("VMOVDQU (AX), Y0").   // acc[0:4]
		Raw("VMOVDQU 32(AX), Y1"). // acc[4:8]
		Label("accumRunAVX2_loop").
		// low half (lanes 0..3)
		Raw("VMOVDQU (CX), Y2").
		Raw("VPXOR (DX), Y2, Y3").
		Raw("VPSHUFD $0x31, Y3, Y4").
		Raw("VPMULUDQ Y3, Y4, Y3").
		Raw("VPSHUFD $0x4e, Y2, Y2").
		Raw("VPADDQ Y0, Y3, Y0").
		Raw("VPADDQ Y0, Y2, Y0").
		// high half (lanes 4..7)
		Raw("VMOVDQU 32(CX), Y5").
		Raw("VPXOR 32(DX), Y5, Y6").
		Raw("VPSHUFD $0x31, Y6, Y7").
		Raw("VPMULUDQ Y6, Y7, Y6").
		Raw("VPSHUFD $0x4e, Y5, Y5").
		Raw("VPADDQ Y1, Y6, Y1").
		Raw("VPADDQ Y1, Y5, Y1").
		// advance pointers: p += 64, sec += 8
		Raw("ADDQ $64, CX").
		Raw("ADDQ $8, DX").
		Raw("DECQ BX").
		Raw("JNZ accumRunAVX2_loop").
		Raw("VMOVDQU Y0, (AX)").
		Raw("VMOVDQU Y1, 32(AX)").
		Raw("VZEROUPPER").
		Ret()
	f.Add(vr.Func())

	// --- AVX2 full-block run: nBlocks complete 1024-byte blocks, each 16 stripes
	// plus the inter-block scramble, all in one asm call. This is the structural
	// match to the canonical/zeebo accumulate_512 loop: the eight accumulator
	// lanes (Y0 lanes 0..3, Y1 lanes 4..7) stay resident in vector registers for
	// the ENTIRE multi-block run (loaded once, stored once) and the per-block
	// scramble runs in-register too, so the long-input pipeline never crosses the
	// Go/asm boundary mid-input. Y8 holds prime32_1 broadcast for the scramble's
	// full 64x32->64 multiply. The 16 stripes are fully unrolled to expose ILP
	// (the two banks' mul/add chains software-pipeline against each other and the
	// next stripe's loads/shuffles start while the current stripe's adds retire);
	// PREFETCHT0 pulls the next cache lines in ahead of use. nBlocks >= 1.
	vb := amd64.NewFunc("accumScrambleAVX2", blockSig(), 0)
	vbb := vb.LoadArg("acc", "AX").LoadArg("p_base", "CX").LoadArg("sec_base", "DX").
		LoadArg("nBlocks", "BX").
		Raw("VMOVDQU (AX), Y0").                   // acc[0:4]
		Raw("VMOVDQU 32(AX), Y1").                 // acc[4:8]
		Raw("VMOVDQU " + primeSym + "+0(SB), Y8"). // prime32_1 broadcast
		Label("accumScrambleAVX2_loop")
	for s := 0; s < 16; s++ {
		dOff := s * 64
		sOff := s * 8
		vbb = vbb.
			Raw("PREFETCHT0 %d(CX)", dOff+512).
			// low half (lanes 0..3)
			Raw("VMOVDQU %d(CX), Y2", dOff).
			Raw("VPXOR %d(DX), Y2, Y3", sOff).
			Raw("VPSHUFD $0x31, Y3, Y4").
			Raw("VPMULUDQ Y3, Y4, Y3").
			Raw("VPSHUFD $0x4e, Y2, Y2").
			Raw("VPADDQ Y0, Y3, Y0").
			Raw("VPADDQ Y0, Y2, Y0").
			// high half (lanes 4..7)
			Raw("VMOVDQU %d(CX), Y5", dOff+32).
			Raw("VPXOR %d(DX), Y5, Y6", sOff+32).
			Raw("VPSHUFD $0x31, Y6, Y7").
			Raw("VPMULUDQ Y6, Y7, Y6").
			Raw("VPSHUFD $0x4e, Y5, Y5").
			Raw("VPADDQ Y1, Y6, Y1").
			Raw("VPADDQ Y1, Y5, Y1")
	}
	// In-register scramble of both banks: v ^= v>>47; v ^= secret[128+i*8];
	// v *= prime32_1 (full 64x32->64 via two PMULUDQ + shift).
	vbb = vbb.
		// low bank Y0
		Raw("VPSRLQ $0x2f, Y0, Y3").
		Raw("VPXOR Y0, Y3, Y3").
		Raw("VPXOR 128(DX), Y3, Y3").
		Raw("VPMULUDQ Y8, Y3, Y0").   // lo32 * prime
		Raw("VPSHUFD $0xf5, Y3, Y3"). // bring hi32 down
		Raw("VPMULUDQ Y8, Y3, Y3").   // hi32 * prime
		Raw("VPSLLQ $0x20, Y3, Y3").
		Raw("VPADDQ Y0, Y3, Y0").
		// high bank Y1
		Raw("VPSRLQ $0x2f, Y1, Y3").
		Raw("VPXOR Y1, Y3, Y3").
		Raw("VPXOR 160(DX), Y3, Y3").
		Raw("VPMULUDQ Y8, Y3, Y1").
		Raw("VPSHUFD $0xf5, Y3, Y3").
		Raw("VPMULUDQ Y8, Y3, Y3").
		Raw("VPSLLQ $0x20, Y3, Y3").
		Raw("VPADDQ Y1, Y3, Y1").
		// next block: p += 1024 (secret base is fixed across blocks)
		Raw("ADDQ $1024, CX").
		Raw("DECQ BX").
		Raw("JNZ accumScrambleAVX2_loop").
		Raw("VMOVDQU Y0, (AX)").
		Raw("VMOVDQU Y1, 32(AX)").
		Raw("VZEROUPPER")
	vbb.Ret()
	f.Add(vb.Func())

	// --- SSE2 single stripe: 8 lanes as four XMM registers (two u64 each). ---
	s := amd64.NewFunc("accumStripeSSE2", stripeSig(), 0)
	sb := s.LoadArg("acc", "AX").LoadArg("p_base", "CX").LoadArg("sec_base", "DX")
	for h := 0; h < 4; h++ {
		off := h * 16
		sb = sb.
			Raw("MOVOU %d(AX), X0", off).
			Raw("MOVOU %d(CX), X1", off). // dv
			Raw("MOVOU %d(DX), X2", off). // sec
			Raw("MOVOU X1, X3").
			Raw("PXOR X2, X3"). // dk = dv ^ sec
			Raw("PSHUFD $0x31, X3, X4").
			Raw("PMULULQ X3, X4").       // X4 = dk.lo * dk.hi  (PMULUDQ)
			Raw("PSHUFD $0x4e, X1, X1"). // swap64(dv)
			Raw("PADDQ X4, X0").
			Raw("PADDQ X1, X0").
			Raw("MOVOU X0, %d(AX)", off)
	}
	sb.Ret()
	f.Add(s.Func())

	// --- SSE2 multi-stripe run: acc resident in X0..X3 across nStripes stripes. ---
	// X0..X3 hold the four lane-pairs (lanes 0..7); X4..X7 are scratch.
	// CX = p, DX = sec, BX = stripe counter.
	sr := amd64.NewFunc("accumRunSSE2", runSig(), 0)
	srb := sr.LoadArg("acc", "AX").LoadArg("p_base", "CX").LoadArg("sec_base", "DX").
		LoadArg("nStripes", "BX").
		Raw("MOVOU (AX), X0").
		Raw("MOVOU 16(AX), X1").
		Raw("MOVOU 32(AX), X2").
		Raw("MOVOU 48(AX), X3").
		Label("accumRunSSE2_loop")
	accReg := []string{"X0", "X1", "X2", "X3"}
	for h := 0; h < 4; h++ {
		off := h * 16
		acc := accReg[h]
		srb = srb.
			Raw("MOVOU %d(CX), X4", off). // dv
			Raw("MOVOU %d(DX), X5", off). // sec
			Raw("MOVOU X4, X6").
			Raw("PXOR X5, X6"). // dk = dv ^ sec
			Raw("PSHUFD $0x31, X6, X7").
			Raw("PMULULQ X6, X7").       // X7 = dk.lo * dk.hi
			Raw("PSHUFD $0x4e, X4, X4"). // swap64(dv)
			Raw("PADDQ X7, " + acc).
			Raw("PADDQ X4, " + acc)
	}
	srb.
		Raw("ADDQ $64, CX").
		Raw("ADDQ $8, DX").
		Raw("DECQ BX").
		Raw("JNZ accumRunSSE2_loop").
		Raw("MOVOU X0, (AX)").
		Raw("MOVOU X1, 16(AX)").
		Raw("MOVOU X2, 32(AX)").
		Raw("MOVOU X3, 48(AX)").
		Ret()
	f.Add(sr.Func())

	// --- SSE2 full-block run: nBlocks complete 1024-byte blocks (16 stripes +
	// scramble) in one asm call, mirroring accumScrambleAVX2 but with the four
	// lane-pairs in X0..X3 resident across the whole run. X8 holds prime32_1
	// broadcast for the scramble multiply; X4..X7 are scratch. nBlocks >= 1.
	sbk := amd64.NewFunc("accumScrambleSSE2", blockSig(), 0)
	sbkb := sbk.LoadArg("acc", "AX").LoadArg("p_base", "CX").LoadArg("sec_base", "DX").
		LoadArg("nBlocks", "BX").
		Raw("MOVOU (AX), X0").
		Raw("MOVOU 16(AX), X1").
		Raw("MOVOU 32(AX), X2").
		Raw("MOVOU 48(AX), X3").
		Raw("MOVOU " + primeSym + "+0(SB), X8").
		Label("accumScrambleSSE2_loop")
	bankReg := []string{"X0", "X1", "X2", "X3"}
	for s := 0; s < 16; s++ {
		dOff := s * 64
		sOff := s * 8
		sbkb = sbkb.Raw("PREFETCHT0 %d(CX)", dOff+512)
		for h := 0; h < 4; h++ {
			acc := bankReg[h]
			sbkb = sbkb.
				Raw("MOVOU %d(CX), X4", dOff+h*16). // dv
				Raw("MOVOU %d(DX), X5", sOff+h*16). // sec
				Raw("MOVOU X4, X6").
				Raw("PXOR X5, X6"). // dk = dv ^ sec
				Raw("PSHUFD $0x31, X6, X7").
				Raw("PMULULQ X6, X7"). // X7 = dk.lo * dk.hi
				Raw("PSHUFD $0x4e, X4, X4").
				Raw("PADDQ X7, " + acc).
				Raw("PADDQ X4, " + acc)
		}
	}
	// In-register scramble of the four banks.
	for h := 0; h < 4; h++ {
		acc := bankReg[h]
		scrOff := 128 + h*16
		sbkb = sbkb.
			Raw("MOVOU "+acc+", X4").
			Raw("PSRLQ $0x2f, X4").
			Raw("PXOR "+acc+", X4").
			Raw("MOVOU %d(DX), X5", scrOff).
			Raw("PXOR X5, X4"). // X4 = (v ^ v>>47) ^ secret
			Raw("MOVOU X4, X6").
			Raw("PMULULQ X8, X6"). // lo32 * prime
			Raw("PSHUFD $0xf5, X4, X4").
			Raw("PMULULQ X8, X4"). // hi32 * prime
			Raw("PSLLQ $0x20, X4").
			Raw("PADDQ X6, X4").
			Raw("MOVOU X4, " + acc)
	}
	sbkb.
		Raw("ADDQ $1024, CX").
		Raw("DECQ BX").
		Raw("JNZ accumScrambleSSE2_loop").
		Raw("MOVOU X0, (AX)").
		Raw("MOVOU X1, 16(AX)").
		Raw("MOVOU X2, 32(AX)").
		Raw("MOVOU X3, 48(AX)").
		Ret()
	f.Add(sbk.Func())

	if err := os.WriteFile("accum_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote accum_amd64.s")
}
