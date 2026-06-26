//go:build ignore

// Command gen produces accum_riscv64.s with go-asmgen: the XXH3 single-stripe
// accumulator on RVV (accumStripeRVV).
//
// One stripe is eight uint64 lanes (VL = 8, SEW = 64). For every lane:
//
//	dk  = dv ^ key
//	lo  = dk & 0xFFFFFFFF          (dk << 32 >> 32)
//	hi  = dk >> 32
//	acc += lo*hi                   (low 64 bits = full product, since lo,hi<2^32)
//	acc += swap-adjacent-lane(dv)  (the acc[i^1] += dv mapping)
//
// The adjacent-lane swap uses a gather with index vid^1 = [1,0,3,2,5,4,7,6].
// Everything is endian-neutral integer add/mul, so the digest matches every
// other architecture. Requires V with VLEN >= 128 (VL=8 at SEW=64 needs 512-bit
// LMUL=4; we set LMUL=M4).
//
// The input stripe (p) and secret (sec) are loaded as byte elements (vle8.v),
// not vle64.v: xxhash feeds arbitrarily-aligned input, and a vle64.v of an
// 8-byte-misaligned address SIGBUSes on hardware lacking misaligned vector
// support (observed on the SpacemiT X60). Reinterpreting the byte-loaded
// register group as u64 lanes is a no-op on this little-endian target.
//
// Run: go run accum_riscv64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/riscv64"
)

func main() {
	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Scalar("acc", abi.Int64), abi.Slice("p"), abi.Slice("sec")},
		[]abi.Arg{},
	)
	b := riscv64.NewFunc("accumStripeRVV", sig, 0)
	b.LoadArg("acc", "X5").LoadArg("p_base", "X6").LoadArg("sec_base", "X7").
		// Load the 64-byte input stripe (p) and secret (sec) as byte elements
		// (vle8.v, vl=64). xxhash feeds arbitrarily-aligned input, and a vle64.v
		// of an 8-byte-misaligned address SIGBUSes on hardware that does not
		// support element-misaligned vector loads (observed on the SpacemiT X60).
		// vle8.v only needs byte alignment, which always holds; reinterpreting the
		// same register group as 8 little-endian u64 lanes is a no-op on a
		// little-endian target, so the digest is unchanged. acc (X5) is a
		// *[8]uint64 and is always 8-byte aligned, so its vle64.v stays.
		Raw("MOV $64, X9").
		Raw("VSETVLI X9, E8, M4, TA, MA, X8").  // 64 byte elements
		Raw("VLE8V (X6), V8").                  // dv bytes (alignment-safe)
		Raw("VLE8V (X7), V12").                 // key bytes (alignment-safe)
		Raw("VSETVLI $8, E64, M4, TA, MA, X8"). // reinterpret: 8 lanes of u64
		Raw("MOV $32, X9").
		Raw("VXORVV V8, V12, V16"). // dk = dv ^ key  (note: VXORVV dst,vs1,vs2)
		// lo = dk & 0xFFFFFFFF via << 32 >> 32 ; hi = dk >> 32
		Raw("VSLLVX X9, V16, V20").
		Raw("VSRLVX X9, V20, V20").  // lo
		Raw("VSRLVX X9, V16, V24").  // hi
		Raw("VMULVV V20, V24, V20"). // lo*hi (full 64-bit product)
		// acc += mul
		Raw("VLE64V (X5), V28").
		Raw("VADDVV V28, V20, V28").
		// swapped dv: idx = vid ^ 1, gather
		Raw("VIDV V4").
		Raw("VXORVI $1, V4, V4").
		Raw("VRGATHERVV V4, V8, V0"). // V0[i] = dv[idx[i]] (idx is V4, data is V8)
		Raw("VADDVV V28, V0, V28").
		Raw("VSE64V V28, (X5)").
		Ret()

	f := emit.NewFile("riscv64")
	f.Add(b.Func())
	if err := os.WriteFile("accum_riscv64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote accum_riscv64.s")
}
