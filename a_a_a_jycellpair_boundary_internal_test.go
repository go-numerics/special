package special

import "testing"

// TestJyCellPairCoefficientBound drives jyCellPair on every cell of the four
// tables and reports a panic as an assertion failure. The Horner loop reads
// rc[0..11] of a [12]float64; a bound of k <= 12 reads rc[12], an
// index-out-of-range panic. A panic alone is luck (mutsweep reads it as
// "caught by PANIC", no test failed by an assertion), so the panic is
// recovered here and reported with t.Errorf, and the file sorts first so the
// assertion is the first red under -failfast.
func TestJyCellPairCoefficientBound(t *testing.T) {
	type table struct {
		name  string
		cells []besselCell
		lut   []uint16
	}
	tables := []table{
		{"J0", besselJ0Cells[:], besselJ0CellsLUT[:]},
		{"J1", besselJ1Cells[:], besselJ1CellsLUT[:]},
		{"Y0", besselY0Cells[:], besselY0CellsLUT[:]},
		{"Y1", besselY1Cells[:], besselY1CellsLUT[:]},
	}
	for _, tb := range tables {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("%s: jyCellPair panicked: %v", tb.name, r)
				}
			}()
			x := tb.cells[0].lo
			if x < 1 {
				x = 1
			}
			jyCellPair(tb.cells, tb.lut, x)
		}()
	}
}
