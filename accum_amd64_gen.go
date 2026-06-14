//go:build ignore

// Command gen produces accum_amd64.s with go-asmgen: the XXH3 single-stripe
// accumulator, both AVX2 (accumStripeAVX2) and SSE2 (accumStripeSSE2).
//
// One stripe is 64 bytes = eight uint64 lanes. For each lane the kernel computes
// dk = dv ^ sec, adds the 32x32->64 product lo32(dk)*hi32(dk) to lane i, and
// adds the input word dv to lane i^1 (the adjacent-lane swap). At 128-bit
// granularity the swap is a within-pair u64 exchange (PSHUFD $0x4e); the product
// uses PSHUFD $0x31 to bring each u64's high half down, then [V]PMULUDQ.
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

func sig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Scalar("acc", abi.Int64), abi.Slice("p"), abi.Slice("sec")},
		[]abi.Arg{},
	)
}

func main() {
	f := emit.NewFile("amd64")

	// AVX2: process all 8 lanes as two YMM registers.
	v := amd64.NewFunc("accumStripeAVX2", sig(), 0)
	v.LoadArg("acc", "AX").LoadArg("p_base", "CX").LoadArg("sec_base", "DX").
		// load accumulator lanes
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
		// store back
		Raw("VMOVDQU Y0, (AX)").
		Raw("VMOVDQU Y1, 32(AX)").
		Raw("VZEROUPPER").
		Ret()
	f.Add(v.Func())

	// SSE2: process the 8 lanes as four XMM registers (two u64 each).
	s := amd64.NewFunc("accumStripeSSE2", sig(), 0)
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

	if err := os.WriteFile("accum_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote accum_amd64.s")
}
