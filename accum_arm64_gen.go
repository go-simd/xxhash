//go:build ignore

// Command gen produces accum_arm64.s with go-asmgen: the XXH3 single-stripe
// accumulator on NEON (accumStripeNEON).
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
// Run: go run accum_arm64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Scalar("acc", abi.Int64), abi.Slice("p"), abi.Slice("sec")},
		[]abi.Arg{},
	)
	b := arm64.NewFunc("accumStripeNEON", sig, 0)
	b.LoadArg("acc", "R0").LoadArg("p_base", "R1").LoadArg("sec_base", "R2").
		// acc pairs: V1=[acc0,acc1] V2=[acc2,acc3] V3=[acc4,acc5] V4=[acc6,acc7]
		Raw("VLD1 (R0), [V1.D2, V2.D2, V3.D2, V4.D2]")

	// XTN  Vd.2S, Vn.2D  -> low 32 of each u64
	xtn := func(d, n int) string { return fmt.Sprintf("WORD $%#x", 0x0EA12800|(n<<5)|d) }
	// SHRN #32, Vn.2D, Vd.2S -> high 32 of each u64
	shrn := func(d, n int) string { return fmt.Sprintf("WORD $%#x", 0x0F208400|(n<<5)|d) }
	// UMULL Vd.2D, Vn.2S, Vm.2S -> 32x32->64
	umull := func(d, n, m int) string { return fmt.Sprintf("WORD $%#x", 0x2EA0C000|(m<<16)|(n<<5)|d) }

	accReg := []string{"V1", "V2", "V3", "V4"}
	for c := 0; c < 4; c++ {
		off := c * 16
		acc := accReg[c]
		b.
			Raw("ADD $%d, R1, R5", off).
			Raw("VLD1 (R5), [V5.D2]"). // dv
			Raw("ADD $%d, R2, R5", off).
			Raw("VLD1 (R5), [V6.D2]").              // key
			Raw("VEOR V5.B16, V6.B16, V6.B16").     // dk = dv ^ key
			Raw(xtn(7, 6)).                         // V7 = lo32(dk)
			Raw(shrn(8, 6)).                        // V8 = hi32(dk)
			Raw(umull(9, 7, 8)).                    // V9 = lo32*hi32
			Raw("VEXT $8, V5.B16, V5.B16, V5.B16"). // swap the pair of dv
			Raw("VADD V5.D2, " + acc + ".D2, " + acc + ".D2").
			Raw("VADD V9.D2, " + acc + ".D2, " + acc + ".D2")
	}
	b.Raw("VST1 [V1.D2, V2.D2, V3.D2, V4.D2], (R0)").Ret()

	f := emit.NewFile("arm64")
	f.Add(b.Func())
	if err := os.WriteFile("accum_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote accum_arm64.s")
}
