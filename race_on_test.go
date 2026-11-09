//go:build race

package special

// raceEnabled: the race detector slows the arithmetic sweeps about tenfold;
// they shrink under it (CI's -race coverage run) and keep their full size
// in the plain run.
const raceEnabled = true
