package conversationstenography

import (
	"context"
	"math"
	"testing"
)

// fixedModel returns the same candidates on every call, so the per-step
// distribution is known exactly and the coder's bits-per-token rate is
// analytically predictable.
type fixedModel struct {
	cands []TokenCandidate
	calls int
}

func (m *fixedModel) Fingerprint() string { return "fixed" }

func (m *fixedModel) Tokenize(context.Context, string) ([]int, error) {
	return []int{1, 2, 3}, nil
}

func (m *fixedModel) Detokenize(context.Context, []int) (string, error) {
	return "fixed text", nil
}

func (m *fixedModel) Next(_ context.Context, _ []int, n int) ([]TokenCandidate, error) {
	m.calls++
	out := make([]TokenCandidate, 0, n)
	for i := 0; i < n && i < len(m.cands); i++ {
		out = append(out, m.cands[i])
	}
	return out, nil
}

// candidateEntropy mirrors makeFrequencies in arithmetic.go and returns the
// Shannon entropy of the resulting distribution in bits.
func candidateEntropy(cands []TokenCandidate, temp, lb float64) float64 {
	maxScore := cands[0].LogProb
	for _, c := range cands[1:] {
		if c.LogProb > maxScore {
			maxScore = c.LogProb
		}
	}
	weights := make([]float64, len(cands))
	sum := 0.0
	for i, c := range cands {
		weights[i] = math.Exp((c.LogProb-maxScore)/temp - lb*float64(len(c.Text)))
		sum += weights[i]
	}
	entropy := 0.0
	for _, w := range weights {
		p := w / sum
		entropy -= p * math.Log2(p)
	}
	return entropy
}

func probeCodec(t *testing.T, cands []TokenCandidate) *GenerativeCodec {
	t.Helper()
	codec, err := NewGenerativeCodec(&fixedModel{cands: cands}, GenerativeConfig{
		Prompt:           "seed",
		TopN:             len(cands),
		Coding:           "arithmetic",
		Temperature:      1.0,
		FinishTokens:     32,
		StrictStyle:      false,
		CandidatePool:    1,
		CarrierTrials:    1,
		NaturalnessSlack: 2,
	})
	if err != nil {
		t.Fatalf("new codec: %v", err)
	}
	return codec
}

// TestArithmeticCoderThroughputTracksEntropy pins the invariant behind the
// carrier length budget at generative.go: the encoder only emits a bit when the
// interval renormalises, so its bits-per-token rate must track the entropy of
// the candidate distribution. A regression that collapses throughput below the
// entropy would silently exhaust the token budget and stall every send.
func TestArithmeticCoderThroughputTracksEntropy(t *testing.T) {
	texts := []string{" alpha", " beta", " gamma", " delta", " epsilon", " zeta", " eta", " theta"}
	for _, spread := range []float64{0.3, 1.0, 2.0} {
		cands := make([]TokenCandidate, 0, len(texts))
		for i, text := range texts {
			cands = append(cands, TokenCandidate{
				ID:      100 + i,
				LogProb: -spread * float64(i),
				Text:    text,
			})
		}
		entropy := candidateEntropy(cands, 1.0, 0)
		codec := probeCodec(t, cands)
		model := codec.model.(*fixedModel)

		payload := make([]byte, 16)
		for i := range payload {
			payload[i] = byte(i * 37)
		}
		// The synthetic model has no real tokenizer, so the final
		// round-trip assertion fails; the token count and the coder's
		// progress are what this probe measures.
		_, _, _ = codec.EncodeUnframedWithMetrics(context.Background(), payload)

		observed := float64(len(payload)*8) / float64(model.calls)
		ratio := observed / entropy
		if ratio < 0.5 {
			t.Errorf("spread %.1f: coder throughput %.4f bits/token is below half the "+
				"candidate entropy %.4f (ratio %.2f); the token budget at generative.go "+
				"assumes at least 0.125 bits/token and would be exhausted early",
				spread, observed, entropy, ratio)
		} else {
			t.Logf("spread %.1f: entropy %.3f, throughput %.3f bits/token (ratio %.2f)",
				spread, entropy, observed, ratio)
		}
	}
}

// TestArithmeticCoderThroughputGrowsWithPayload records that throughput is not a
// fixed constant: the encoder primes with 32 bits before any symbol, so short
// payloads amortise that badly. Larger payloads must reach a higher rate.
func TestArithmeticCoderThroughputGrowsWithPayload(t *testing.T) {
	texts := []string{" alpha", " beta", " gamma", " delta", " epsilon", " zeta", " eta", " theta"}
	cands := make([]TokenCandidate, 0, len(texts))
	for i, text := range texts {
		cands = append(cands, TokenCandidate{ID: 100 + i, LogProb: -1.0 * float64(i), Text: text})
	}
	rates := make(map[int]float64)
	for _, size := range []int{4, 8, 16, 32, 64} {
		codec := probeCodec(t, cands)
		model := codec.model.(*fixedModel)
		payload := make([]byte, size)
		_, _, _ = codec.EncodeUnframedWithMetrics(context.Background(), payload)
		rates[size] = float64(size*8) / float64(model.calls)
		t.Logf("payload %3dB -> %.4f bits/token over %d tokens", size, rates[size], model.calls)
	}
	if rates[64] <= rates[4] {
		t.Errorf("throughput did not improve with payload size: 4B=%.4f 64B=%.4f",
			rates[4], rates[64])
	}
}
