package conversationstenography

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/binary"
	"errors"
)

// SymbolMap assigns a compact payload symbol to a visible token.
//
// The mapping is deliberately independent of the carrier language model.
// A sender may use any generator capable of proposing tokens accepted by the
// map; a receiver only needs the canonical tokenizer, the map definition, and
// the shared key.
//
// Implementations must be deterministic for the same key, visible token
// history, and token ID.
type SymbolMap interface {
	BitsPerSymbol() int
	Symbol(key []byte, visibleContext []int, tokenID int) (uint64, error)
}

// PRFSymbolMap is the bootstrap codebook.
//
// It does not claim semantic robustness: it partitions token IDs with a keyed
// HMAC over the visible token history. Its purpose is to prove the architecture
// in which generator synchronization is no longer part of decoding. A future
// semantic map can replace it without changing framing or encryption.
type PRFSymbolMap struct {
	bits int
}

// NewPRFSymbolMap creates a keyed token partition carrying bitsPerSymbol bits
// per selected token. Keeping the bootstrap map at <= 8 bits bounds rejection
// search costs and prevents accidental use as a high-capacity construction.
func NewPRFSymbolMap(bitsPerSymbol int) (*PRFSymbolMap, error) {
	if bitsPerSymbol < 1 || bitsPerSymbol > 8 {
		return nil, errors.New("bits per symbol must be between 1 and 8")
	}
	return &PRFSymbolMap{bits: bitsPerSymbol}, nil
}

func (m *PRFSymbolMap) BitsPerSymbol() int { return m.bits }

func (m *PRFSymbolMap) Symbol(key []byte, visibleContext []int, tokenID int) (uint64, error) {
	if len(key) < 16 {
		return 0, errors.New("symbol-map key must contain at least 16 bytes of entropy")
	}

	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte("llm-steg-symbol-map/prf/v1\x00"))

	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], uint64(len(visibleContext)))
	_, _ = mac.Write(buf[:])

	for _, id := range visibleContext {
		binary.BigEndian.PutUint64(buf[:], uint64(int64(id)))
		_, _ = mac.Write(buf[:])
	}

	binary.BigEndian.PutUint64(buf[:], uint64(int64(tokenID)))
	_, _ = mac.Write(buf[:])

	sum := mac.Sum(nil)
	mask := uint64((1 << m.bits) - 1)
	return uint64(sum[0]) & mask, nil
}
