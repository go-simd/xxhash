package xxhash

import (
	"fmt"
	"testing"
)

var benchSizes = []int{8, 64, 256, 1024, 8192, 65536}

func BenchmarkSum64(b *testing.B) {
	for _, n := range benchSizes {
		buf := make([]byte, n)
		for i := range buf {
			buf[i] = byte(i)
		}
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			var s uint64
			for i := 0; i < b.N; i++ {
				s += Sum64(buf)
			}
			_ = s
		})
	}
}

func BenchmarkStreaming(b *testing.B) {
	for _, n := range []int{1024, 65536} {
		buf := make([]byte, n)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			d := New()
			for i := 0; i < b.N; i++ {
				d.Reset()
				_, _ = d.Write(buf)
				_ = d.Sum64()
			}
		})
	}
}
