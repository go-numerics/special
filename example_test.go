package special_test

import (
	"fmt"

	"github.com/go-numerics/special"
)

func Example() {
	fmt.Println(special.Gamma(4.5))
	fmt.Println(special.Erfc(10))              // 2.088487583762545e-45, no underflow to 0
	fmt.Println(special.Ndtr(-30))             // the far tail keeps full relative accuracy
	fmt.Println(special.Ndtri(1e-300))         // and so does its inverse
	fmt.Println(special.LogGamma(1 + 0x1p-30)) // log Γ near its zero at 1
	// Output:
	// 11.631728396567448
	// 2.088487583762545e-45
	// 4.906713927148187e-198
	// -37.0470962993612
	// -5.375739784311044e-10
}
