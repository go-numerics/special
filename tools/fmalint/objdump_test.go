package main

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestImplicitFusions(t *testing.T) {
	root := t.TempDir()
	src := "package q\n\nfunc a(x, y, z float64) float64 {\n\treturn x*y + z\n}\n\nfunc b(x, y, z float64) float64 {\n\treturn math.FMA(x, y, z)\n}\n"
	if err := os.WriteFile(filepath.Join(root, "q.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(root, "q_test.go"), []byte(src), 0o644)
	dump := strings.Join([]string{
		"TEXT example.com/q.a(SB) " + filepath.Join(root, "q.go"),
		"  q.go:4\t0x1000\t\t1f420020\t\tFMADDD F2, F0, F1, F0\t",
		"TEXT example.com/q.b(SB) " + filepath.Join(root, "q.go"),
		"  q.go:8\t0x1010\t\t1f420020\t\tFMADDD F2, F0, F1, F0\t",
		"  q_test.go:4\t0x1014\t\t1f420020\t\tFMADDD F2, F0, F1, F0\t",
		"TEXT runtime.x(SB) /goroot/x.go",
		"  x.go:1\t0x2000\t\t1f420020\t\tFMADDD F2, F0, F1, F0\t",
	}, "\n")
	got, err := ImplicitFusions(strings.NewReader(dump), "example.com/q", root)
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"q.go:4"}; !reflect.DeepEqual(got, want) {
		t.Errorf("ImplicitFusions = %v, want %v", got, want)
	}
}
