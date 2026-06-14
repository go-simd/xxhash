//go:build ppc64le

package xxhash

// accumStripeVSX folds one 64-byte stripe into the eight lanes of acc using VSX.
// VSX is baseline on POWER8+, so there is no runtime feature dispatch. Generated
// by go-asmgen (accum_ppc64le_gen.go) into accum_ppc64le.s.
func accumStripeVSX(acc *[8]uint64, p, sec []byte)

func accumStripe(acc *[8]uint64, p, sec []byte) {
	accumStripeVSX(acc, p, sec)
}
