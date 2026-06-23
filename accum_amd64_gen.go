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

func main() {
	f := emit.NewFile("amd64")

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

	if err := os.WriteFile("accum_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote accum_amd64.s")
}
