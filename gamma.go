package special

import "math"

// Gamma returns the gamma function Γ(x). (placeholder: being replaced)
func Gamma(x float64) float64 { return math.Gamma(x) }

// LogGamma returns log|Γ(x)|. (placeholder: being replaced)
func LogGamma(x float64) float64 { v, _ := math.Lgamma(x); return v }

// LogGammaSign returns log|Γ(x)| and the sign of Γ(x). (placeholder)
func LogGammaSign(x float64) (float64, int) { return math.Lgamma(x) }
