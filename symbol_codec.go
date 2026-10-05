package conversationstenography

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
)

// TextTokenizer is the receiver-side surface required by the keyed token
// codec. In particular, decoding does not require next-token logits or the
// carrier generator's weights.
type TextTokenizer interface {
	Tokenize(context.Context, string) ([]int, error)
	Detokenize(context.Context, []int) (string, error)
}

// KeyedTokenConfig controls sender-side carrier generation. Prompt and
// CandidatePool are deliberately local generation concerns; the receiver does
// not need either value to decode the hidden payload.
type KeyedTokenConfig struct {
	Prompt        string
	CandidatePool int
}

// KeyedTokenEncoder selects the highest-ranked candidate whose keyed symbol
// matches the next payload symbol. The language model supplies plausibility;
// SymbolMap supplies protocol meaning.
type KeyedTokenEncoder struct {
	model   LanguageModel
	symbols SymbolMap
	key     []byte
	cfg     KeyedTokenConfig
}

// KeyedTokenDecoder reconstructs symbols from visible token IDs. It has no
// language-model dependency by design.
type KeyedTokenDecoder struct {
	tokenizer TextTokenizer
	symbols   SymbolMap
	key       []byte
}

func NewKeyedTokenEncoder(model LanguageModel, symbols SymbolMap, key []byte, cfg KeyedTokenConfig) (*KeyedTokenEncoder, error) {
	if model == nil {
		return nil, errors.New("carrier generator is required")
	}
	if symbols == nil {
		return nil, errors.New("symbol map is required")
	}
	if len(key) < 16 {
		return nil, errors.New("keyed token codec requires at least 16 bytes of key material")
	}
	if cfg.Prompt == "" {
		return nil, errors.New("carrier prompt must not be empty")
	}
	if cfg.CandidatePool < 2 {
		return nil, errors.New("candidate pool must be at least 2")
	}
	bits := symbols.BitsPerSymbol()
	if bits < 1 || bits > 8 {
		return nil, errors.New("symbol map must carry between 1 and 8 bits per token")
	}
	return &KeyedTokenEncoder{model: model, symbols: symbols, key: append([]byte(nil), key...), cfg: cfg}, nil
}

func NewKeyedTokenDecoder(tokenizer TextTokenizer, symbols SymbolMap, key []byte) (*KeyedTokenDecoder, error) {
	if tokenizer == nil {
		return nil, errors.New("canonical tokenizer is required")
	}
	if symbols == nil {
		return nil, errors.New("symbol map is required")
	}
	if len(key) < 16 {
		return nil, errors.New("keyed token codec requires at least 16 bytes of key material")
	}
	bits := symbols.BitsPerSymbol()
	if bits < 1 || bits > 8 {
		return nil, errors.New("symbol map must carry between 1 and 8 bits per token")
	}
	return &KeyedTokenDecoder{tokenizer: tokenizer, symbols: symbols, key: append([]byte(nil), key...)}, nil
}

// Encode frames payload with a uvarint byte length, then hides the framed bits
// in keyed token classes. No receiver-side model synchronization is required.
func (e *KeyedTokenEncoder) Encode(ctx context.Context, payload []byte) (string, error) {
	frame := binary.AppendUvarint(nil, uint64(len(payload)))
	frame = append(frame, payload...)

	modelContext, err := e.model.Tokenize(ctx, e.cfg.Prompt)
	if err != nil {
		return "", fmt.Errorf("tokenize carrier prompt: %w", err)
	}

	bitsPerSymbol := e.symbols.BitsPerSymbol()
	visible := make([]int, 0, (len(frame)*8+bitsPerSymbol-1)/bitsPerSymbol)

	for bitOffset := 0; bitOffset < len(frame)*8; bitOffset += bitsPerSymbol {
		target := readKeyedSymbol(frame, bitOffset, bitsPerSymbol)
		candidates, err := e.nextCandidates(ctx, modelContext, visible)
		if err != nil {
			return "", err
		}

		selected := -1
		for _, candidate := range candidates {
			symbol, err := e.symbols.Symbol(e.key, visible, candidate.ID)
			if err != nil {
				return "", fmt.Errorf("classify carrier candidate: %w", err)
			}
			if symbol == target {
				selected = candidate.ID
				break
			}
		}
		if selected < 0 {
			return "", fmt.Errorf("candidate pool of %d contains no token for symbol %d at carrier position %d", len(candidates), target, len(visible))
		}

		visible = append(visible, selected)
		modelContext = append(modelContext, selected)
	}

	text, err := e.model.Detokenize(ctx, visible)
	if err != nil {
		return "", fmt.Errorf("detokenize carrier: %w", err)
	}
	roundTrip, err := e.model.Tokenize(ctx, text)
	if err != nil {
		return "", fmt.Errorf("validate carrier tokenization: %w", err)
	}
	if !sameTokenIDs(visible, roundTrip) {
		return "", errors.New("canonical tokenizer cannot losslessly represent generated carrier")
	}
	return text, nil
}

func (e *KeyedTokenEncoder) nextCandidates(ctx context.Context, modelContext, visible []int) ([]TokenCandidate, error) {
	var (
		candidates []TokenCandidate
		err        error
	)
	if safe, ok := e.model.(copySafeLanguageModel); ok {
		candidates, err = safe.NextCopySafe(ctx, modelContext, visible, e.cfg.CandidatePool)
	} else {
		candidates, err = e.model.Next(ctx, modelContext, e.cfg.CandidatePool)
	}
	if err != nil {
		return nil, fmt.Errorf("generate carrier candidates: %w", err)
	}
	if len(candidates) == 0 {
		return nil, errors.New("carrier generator returned no candidates")
	}
	return candidates, nil
}

// Decode reconstructs a framed payload using only visible token IDs and the
// keyed SymbolMap. Trailing carrier text after a complete frame is ignored so
// a future sender can add unencoded sentence-finishing text without restoring
// a receiver-side generator dependency.
func (d *KeyedTokenDecoder) Decode(ctx context.Context, carrier string) ([]byte, error) {
	observed, err := d.tokenizer.Tokenize(ctx, carrier)
	if err != nil {
		return nil, fmt.Errorf("tokenize carrier: %w", err)
	}
	if len(observed) == 0 {
		return nil, errors.New("carrier contains no tokens")
	}

	bitsPerSymbol := d.symbols.BitsPerSymbol()
	decoded := make([]byte, 0, len(observed)*bitsPerSymbol/8+1)
	bitOffset := 0
	visible := make([]int, 0, len(observed))

	for _, tokenID := range observed {
		symbol, err := d.symbols.Symbol(d.key, visible, tokenID)
		if err != nil {
			return nil, fmt.Errorf("decode carrier symbol: %w", err)
		}
		appendKeyedSymbol(&decoded, &bitOffset, symbol, bitsPerSymbol)
		visible = append(visible, tokenID)

		completeBytes := bitOffset / 8
		if completeBytes == 0 {
			continue
		}
		length, headerBytes := binary.Uvarint(decoded[:completeBytes])
		if headerBytes < 0 {
			return nil, errors.New("carrier contains an invalid length frame")
		}
		if headerBytes == 0 {
			continue
		}
		if length > uint64(^uint(0)>>1) {
			return nil, errors.New("carrier declares an unsupported payload length")
		}
		requiredBytes := headerBytes + int(length)
		requiredBits := requiredBytes * 8
		if bitOffset < requiredBits {
			continue
		}

		for i := requiredBits; i < bitOffset; i++ {
			if keyedBit(decoded, i) != 0 {
				return nil, errors.New("carrier has non-zero symbol padding")
			}
		}
		if len(decoded) < requiredBytes {
			return nil, errors.New("carrier ended before the declared payload")
		}
		payload := append([]byte(nil), decoded[headerBytes:requiredBytes]...)
		return payload, nil
	}

	return nil, errors.New("carrier ended before a complete payload frame was decoded")
}

func readKeyedSymbol(data []byte, bitOffset, width int) uint64 {
	var out uint64
	for i := 0; i < width; i++ {
		out <<= 1
		pos := bitOffset + i
		if pos >= len(data)*8 {
			continue
		}
		if data[pos/8]&(1<<uint(7-pos%8)) != 0 {
			out |= 1
		}
	}
	return out
}

func appendKeyedSymbol(dst *[]byte, bitOffset *int, symbol uint64, width int) {
	for i := width - 1; i >= 0; i-- {
		byteIndex := *bitOffset / 8
		if byteIndex >= len(*dst) {
			*dst = append(*dst, 0)
		}
		if (symbol>>uint(i))&1 != 0 {
			(*dst)[byteIndex] |= 1 << uint(7-*bitOffset%8)
		}
		*bitOffset++
	}
}

func keyedBit(data []byte, offset int) int {
	if offset < 0 || offset >= len(data)*8 {
		return 0
	}
	if data[offset/8]&(1<<uint(7-offset%8)) != 0 {
		return 1
	}
	return 0
}

func sameTokenIDs(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
