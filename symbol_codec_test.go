package conversationstenography

import (
	"context"
	"fmt"
	"testing"
)

type moduloSymbolMap struct{ bits int }

func (m moduloSymbolMap) BitsPerSymbol() int { return m.bits }
func (m moduloSymbolMap) Symbol(_ []byte, _ []int, tokenID int) (uint64, error) {
	return uint64(tokenID) & uint64((1<<m.bits)-1), nil
}

type testGenerator struct {
	order []int
}

func (m *testGenerator) Fingerprint() string { return "test-generator" }
func (m *testGenerator) Tokenize(_ context.Context, text string) ([]int, error) {
	out := make([]int, 0, len(text))
	for _, r := range text {
		if r < 'a' || r > 'p' {
			return nil, fmt.Errorf("unsupported rune %q", r)
		}
		out = append(out, int(r-'a'))
	}
	return out, nil
}
func (m *testGenerator) Detokenize(_ context.Context, tokens []int) (string, error) {
	runes := make([]rune, len(tokens))
	for i, id := range tokens {
		if id < 0 || id > 15 {
			return "", fmt.Errorf("unsupported token %d", id)
		}
		runes[i] = rune('a' + id)
	}
	return string(runes), nil
}
func (m *testGenerator) Next(_ context.Context, _ []int, topN int) ([]TokenCandidate, error) {
	order := m.order
	if len(order) == 0 {
		order = []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	}
	if topN > len(order) {
		topN = len(order)
	}
	out := make([]TokenCandidate, topN)
	for i, id := range order[:topN] {
		out[i] = TokenCandidate{ID: id, LogProb: -float64(i), Text: string(rune('a' + id))}
	}
	return out, nil
}

type tokenizerOnly struct{}

func (tokenizerOnly) Tokenize(_ context.Context, text string) ([]int, error) {
	return (&testGenerator{}).Tokenize(context.Background(), text)
}
func (tokenizerOnly) Detokenize(_ context.Context, tokens []int) (string, error) {
	return (&testGenerator{}).Detokenize(context.Background(), tokens)
}

func TestKeyedTokenCodecReceiverDoesNotNeedGenerator(t *testing.T) {
	key := []byte("0123456789abcdef0123456789abcdef")
	symbols := moduloSymbolMap{bits: 2}
	encoder, err := NewKeyedTokenEncoder(&testGenerator{}, symbols, key, KeyedTokenConfig{
		Prompt: "a", CandidatePool: 8,
	})
	if err != nil {
		t.Fatal(err)
	}
	decoder, err := NewKeyedTokenDecoder(tokenizerOnly{}, symbols, key)
	if err != nil {
		t.Fatal(err)
	}

	want := []byte("generator independent")
	carrier, err := encoder.Encode(context.Background(), want)
	if err != nil {
		t.Fatal(err)
	}
	got, err := decoder.Decode(context.Background(), carrier)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Fatalf("decoded %q; want %q", got, want)
	}
}

func TestKeyedTokenCodecUsesGeneratorRankingOnlyForPlausibility(t *testing.T) {
	key := []byte("0123456789abcdef0123456789abcdef")
	symbols := moduloSymbolMap{bits: 1}
	first := &testGenerator{order: []int{0, 2, 4, 6, 1, 3, 5, 7}}
	second := &testGenerator{order: []int{7, 5, 3, 1, 6, 4, 2, 0}}

	encA, _ := NewKeyedTokenEncoder(first, symbols, key, KeyedTokenConfig{Prompt: "a", CandidatePool: 8})
	encB, _ := NewKeyedTokenEncoder(second, symbols, key, KeyedTokenConfig{Prompt: "a", CandidatePool: 8})
	dec, _ := NewKeyedTokenDecoder(tokenizerOnly{}, symbols, key)

	payload := []byte("same payload")
	a, err := encA.Encode(context.Background(), payload)
	if err != nil {
		t.Fatal(err)
	}
	b, err := encB.Encode(context.Background(), payload)
	if err != nil {
		t.Fatal(err)
	}
	if a == b {
		t.Fatal("different generator rankings unexpectedly produced identical carriers")
	}
	for _, carrier := range []string{a, b} {
		got, err := dec.Decode(context.Background(), carrier)
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != string(payload) {
			t.Fatalf("decoded %q; want %q", got, payload)
		}
	}
}

func TestPRFSymbolMapDeterministicAndBounded(t *testing.T) {
	m, err := NewPRFSymbolMap(4)
	if err != nil {
		t.Fatal(err)
	}
	key := []byte("0123456789abcdef0123456789abcdef")
	contextIDs := []int{12, 99, 5}
	a, err := m.Symbol(key, contextIDs, 42)
	if err != nil {
		t.Fatal(err)
	}
	b, err := m.Symbol(key, contextIDs, 42)
	if err != nil {
		t.Fatal(err)
	}
	if a != b {
		t.Fatalf("symbol map is not deterministic: %d != %d", a, b)
	}
	if a >= 16 {
		t.Fatalf("4-bit symbol out of range: %d", a)
	}
}
