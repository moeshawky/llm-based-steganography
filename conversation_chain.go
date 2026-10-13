package conversationstenography

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strings"
)

// ChainRecord is the public state needed by every member of a group. Plaintext
// is deliberately absent; peers may keep different local views of decrypted
// messages while sharing the same ordered carrier transcript.
type ChainRecord struct {
	Index          uint64 `json:"index"`
	From           string `json:"from"`
	SenderSequence uint64 `json:"sender_sequence"`
	Encrypted      string `json:"encrypted"`
}

// ConversationChain turns a sequence of messages from multiple senders into
// one authenticated, ordered, rolling generative conversation.
type ConversationChain struct {
	model        LanguageModel
	key          []byte
	conversation string
	baseConfig   GenerativeConfig
	records      []ChainRecord
	sequences    map[string]uint64
	chain        [32]byte
}

// EncodingBudget accounts for every bit embedded in a text carrier. Prompt,
// sender, sequence, and chain state are synchronized context and are not part
// of the encoded payload.
type EncodingBudget struct {
	PlaintextBytes                   int
	PackedBytes                      int
	AuthenticationBytes              int
	FrameLengthBytes                 int
	TerminationBytes                 int
	TotalHiddenBytes                 int
	BaselineAuthenticationHypotheses int
	BaselineAuthenticationBits       float64
}

const maxAuthenticationHypotheses = 1 << 15

func authenticationHypotheses(candidateCount, trials int) (int, error) {
	if candidateCount < 1 || trials < 1 {
		return 0, errors.New("authentication search dimensions must be positive")
	}
	// Each boundary candidate tries every detached mode and trial, then the
	// inline-mode legacy form and the former GCM form.
	perCandidate := trials*(len(packingModes())+1) + 1
	if candidateCount > maxAuthenticationHypotheses/perCandidate {
		return 0, fmt.Errorf("authentication search would exceed %d hypotheses", maxAuthenticationHypotheses)
	}
	return candidateCount * perCandidate, nil
}

func effectiveAuthenticationBits(hypotheses int) float64 {
	if hypotheses < 1 {
		return 0
	}
	return 128 - math.Log2(float64(hypotheses))
}

func (c *ConversationChain) EncodingBudget(plaintext []byte) (EncodingBudget, error) {
	_, packed, err := packMessageDetached(plaintext, c.compressionDictionary())
	if err != nil {
		return EncodingBudget{}, err
	}
	authenticationBytes := sivTagSize
	frameLengthBytes := 0
	trials := c.baseConfig.CarrierTrials
	if trials == 0 {
		trials = 1
	}
	hypotheses, err := authenticationHypotheses(1, trials)
	if err != nil {
		return EncodingBudget{}, err
	}
	authenticationBits := effectiveAuthenticationBits(hypotheses)
	sealedBytes := authenticationBytes + len(packed)
	if !c.usesUnframedArithmetic() {
		frameLengthBytes = binary.PutUvarint(make([]byte, binary.MaxVarintLen64), uint64(sealedBytes))
	}
	return EncodingBudget{
		PlaintextBytes: len(plaintext), PackedBytes: len(packed), AuthenticationBytes: authenticationBytes,
		FrameLengthBytes: frameLengthBytes, TerminationBytes: 0,
		TotalHiddenBytes:                 frameLengthBytes + sealedBytes,
		BaselineAuthenticationHypotheses: hypotheses, BaselineAuthenticationBits: authenticationBits,
	}, nil
}

func NewConversationChain(model LanguageModel, key []byte, conversation string, cfg GenerativeConfig) (*ConversationChain, error) {
	if model == nil {
		return nil, errors.New("language model is required")
	}
	if len(key) < 16 {
		return nil, errors.New("chain key must contain at least 16 bytes of entropy")
	}
	if strings.TrimSpace(conversation) == "" {
		return nil, errors.New("conversation identifier is required")
	}
	seed := sha256.Sum256([]byte("decalgo-group-chain-v1\x00" + conversation))
	return &ConversationChain{model: model, key: append([]byte(nil), key...), conversation: conversation,
		baseConfig: cfg, sequences: make(map[string]uint64), chain: seed}, nil
}

func (c *ConversationChain) Records() []ChainRecord { return append([]ChainRecord(nil), c.records...) }

func (c *ConversationChain) SyncCode() string {
	return fmt.Sprintf("%x", c.chain[:6])
}

// SetTokenCallback installs fn as the live-streaming callback for all
// subsequent Send calls. Each generated token's text is forwarded to fn as
// it is selected, giving callers real-time visibility into carrier generation.
// Pass nil to disable streaming.
func (c *ConversationChain) SetTokenCallback(fn func(string)) {
	c.baseConfig.TokenCallback = fn
}

// RestorePublic rebuilds rolling prompt, index, sender counters, and hash state
// from a previously persisted public transcript.
func (c *ConversationChain) RestorePublic(records []ChainRecord) error {
	for _, record := range records {
		if record.Index != uint64(len(c.records)) {
			return fmt.Errorf("record index %d; want %d", record.Index, len(c.records))
		}
		wantSequence := c.sequences[record.From]
		if record.SenderSequence != wantSequence {
			return fmt.Errorf("sender %q sequence %d; want %d", record.From, record.SenderSequence, wantSequence)
		}
		if err := validateSender(record.From); err != nil {
			return err
		}
		if record.Encrypted == "" {
			return errors.New("empty encrypted carrier in chain state")
		}
		c.commit(record)
	}
	return nil
}

func (c *ConversationChain) Send(ctx context.Context, from string, plaintext []byte) (ChainRecord, error) {
	if err := validateSender(from); err != nil {
		return ChainRecord{}, err
	}
	record := ChainRecord{Index: uint64(len(c.records)), From: from, SenderSequence: c.sequences[from]}
	codec, err := NewGenerativeCodec(c.model, c.messageConfig(from))
	if err != nil {
		return ChainRecord{}, err
	}
	trialCount := codec.Config().CarrierTrials
	var options []carrierOption
	var lastErr error
	for trial := 0; trial < trialCount; trial++ {
		sealed, err := c.sealTrial(from, record.Index, record.SenderSequence, trial, plaintext)
		if err != nil {
			return ChainRecord{}, err
		}
		var carrier string
		var metrics CarrierMetrics
		if c.usesUnframedArithmetic() {
			carrier, metrics, err = codec.EncodeUnframedWithMetrics(ctx, sealed)
		} else {
			carrier, metrics, err = codec.EncodeWithMetrics(ctx, sealed)
		}
		if err != nil {
			lastErr = err
			if strings.Contains(err.Error(), "tokenizer cannot losslessly represent") {
				continue
			}
			return ChainRecord{}, err
		}
		human, reason := humanWrittenCarrierReason(carrier)
		semanticMargin := math.Inf(1)
		if codec.Config().SemanticJudge {
			semanticMargin = math.Inf(-1)
		}
		if human && codec.Config().SemanticJudge {
			_, semanticMargin, err = semanticHumanWritten(ctx, c.model, carrier)
			if err != nil {
				return ChainRecord{}, fmt.Errorf("judge generated carrier: %w", err)
			}
			human = semanticMargin >= codec.Config().SemanticThreshold
			if !human {
				reason = fmt.Sprintf("semantic judge scored margin %.3f, below threshold %.3f", semanticMargin, codec.Config().SemanticThreshold)
			}
		}
		options = append(options, carrierOption{text: carrier, metrics: metrics, human: human, semanticMargin: semanticMargin, reason: reason})
	}
	best := selectCarrier(options, codec.Config().StrictStyle, codec.Config().NaturalnessSlack)
	if best == "" {
		if lastErr != nil {
			return ChainRecord{}, fmt.Errorf("could not produce a transport-safe carrier after %d trials: %w", trialCount, lastErr)
		}
		if len(options) != 0 {
			reasons := make([]string, 0, len(options))
			for _, option := range options {
				reasons = append(reasons, option.reason)
			}
			return ChainRecord{}, fmt.Errorf("none of %d carrier trials passed the human-writing checks: %s", trialCount, strings.Join(reasons, "; "))
		}
		return ChainRecord{}, fmt.Errorf("none of %d carrier trials produced a carrier", trialCount)
	}
	record.Encrypted = best
	c.commit(record)
	return record, nil
}

type carrierOption struct {
	text           string
	metrics        CarrierMetrics
	human          bool
	semanticMargin float64
	reason         string
}

func (o carrierOption) naturalnessScore() float64 {
	score := o.metrics.MeanLogitRegret + 0.05*o.metrics.WorstLogitRegret
	if !math.IsInf(o.semanticMargin, 0) {
		score -= 0.1 * o.semanticMargin
	}
	return score
}

func selectCarrier(options []carrierOption, strict bool, slack float64) string {
	if !strict {
		slack = math.Inf(1)
	}
	bestNaturalness := math.Inf(1)
	for _, option := range options {
		if strict && !option.human {
			continue
		}
		if score := option.naturalnessScore(); score < bestNaturalness {
			bestNaturalness = score
		}
	}
	best := ""
	bestLength := 0
	for _, option := range options {
		if strict && !option.human {
			continue
		}
		if option.naturalnessScore() > bestNaturalness+slack {
			continue
		}
		length := option.metrics.VisibleCharacters
		if best == "" || length < bestLength {
			best, bestLength = option.text, length
		}
	}
	return best
}

func (c *ConversationChain) Receive(ctx context.Context, from, encrypted string) ([]byte, ChainRecord, error) {
	if err := validateSender(from); err != nil {
		return nil, ChainRecord{}, err
	}
	if encrypted == "" {
		return nil, ChainRecord{}, errors.New("encrypted carrier is empty")
	}
	record := ChainRecord{Index: uint64(len(c.records)), From: from, SenderSequence: c.sequences[from], Encrypted: encrypted}
	codec, err := NewGenerativeCodec(c.model, c.messageConfig(from))
	if err != nil {
		return nil, ChainRecord{}, err
	}
	var plaintext []byte
	accepted := false
	if c.usesUnframedArithmetic() {
		var candidates [][]byte
		candidates, err = codec.DecodeUnframedCandidates(ctx, encrypted)
		if err == nil {
			if _, searchErr := authenticationHypotheses(len(candidates), codec.Config().CarrierTrials); searchErr != nil {
				return nil, ChainRecord{}, searchErr
			}
			for _, sealed := range candidates {
				plaintext, err = c.openTrials(from, record.Index, record.SenderSequence, sealed, codec.Config().CarrierTrials)
				if err == nil {
					accepted = true
					break
				}
			}
		}
	}
	if !accepted {
		var sealed []byte
		sealed, err = codec.Decode(ctx, encrypted)
		if err == nil {
			plaintext, err = c.openTrials(from, record.Index, record.SenderSequence, sealed, codec.Config().CarrierTrials)
			accepted = err == nil
		}
	}
	if !accepted && err == nil {
		err = errors.New("carrier did not contain an authentic payload")
	}
	if err != nil {
		return nil, ChainRecord{}, fmt.Errorf("%w at index %d for sender %q (local sync %s); verify the same phrase and conversation, exact sender name, exact unedited carrier, and identical prior-message order on both devices", err, record.Index, from, c.SyncCode())
	}
	c.commit(record)
	return plaintext, record, nil
}

func (c *ConversationChain) usesUnframedArithmetic() bool {
	return c.baseConfig.Coding == "arithmetic" && !c.baseConfig.RefreshSentences
}

func (c *ConversationChain) messageConfig(from string) GenerativeConfig {
	cfg := c.baseConfig
	if cfg.ChainSystem != "" {
		var transcript strings.Builder
		if len(c.records) == 0 {
			transcript.WriteString("The group chat has just started.")
		} else {
			transcript.WriteString("Earlier chat messages, oldest first:\n\n")
			for _, record := range c.records {
				transcript.WriteString(escapePromptControl(record.Encrypted))
				transcript.WriteString("\n\n")
			}
		}
		transcript.WriteString("\nWrite only one natural reply by the current participant. Do not include a name, label, signature, or transcript.")
		cfg.Prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + cfg.ChainSystem +
			"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + transcript.String() +
			"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
		return cfg
	}
	var prompt strings.Builder
	prompt.WriteString(cfg.Prompt)
	prompt.WriteString("\n\nVisible group conversation:\n")
	for _, record := range c.records {
		prompt.WriteString(record.Encrypted)
		prompt.WriteString("\n\n")
	}
	prompt.WriteString("Write only one natural unlabeled reply: ")
	cfg.Prompt = prompt.String()
	return cfg
}

func escapePromptControl(text string) string { return strings.ReplaceAll(text, "<|", "< |") }

func (c *ConversationChain) commit(record ChainRecord) {
	h := sha256.New()
	h.Write([]byte("decalgo-group-commit-v1\x00"))
	h.Write(c.chain[:])
	h.Write(chainNumbers(record.Index, record.SenderSequence))
	h.Write([]byte(record.From))
	h.Write([]byte{0})
	h.Write([]byte(record.Encrypted))
	copy(c.chain[:], h.Sum(nil))
	c.records = append(c.records, record)
	c.sequences[record.From]++
}

func (c *ConversationChain) seal(from string, index, sequence uint64, plaintext []byte) ([]byte, error) {
	return c.sealTrial(from, index, sequence, 0, plaintext)
}

func (c *ConversationChain) sealTrial(from string, index, sequence uint64, trial int, plaintext []byte) ([]byte, error) {
	mode, packed, err := packMessageDetached(plaintext, c.compressionDictionary())
	if err != nil {
		return nil, fmt.Errorf("pack chained message: %w", err)
	}
	return sealSIV(c.key, c.trialModeAAD(from, index, sequence, trial, mode), packed)
}

func (c *ConversationChain) open(from string, index, sequence uint64, sealed []byte) ([]byte, error) {
	return c.openTrials(from, index, sequence, sealed, 1)
}

func (c *ConversationChain) openTrials(from string, index, sequence uint64, sealed []byte, trials int) ([]byte, error) {
	var packed []byte
	var err error
	for trial := 0; trial < trials; trial++ {
		for _, mode := range packingModes() {
			var body []byte
			body, err = openSIV(c.key, c.trialModeAAD(from, index, sequence, trial, mode), sealed)
			if err == nil {
				packed = append([]byte{mode}, body...)
				break
			}
		}
		if packed != nil {
			break
		}
	}
	if err != nil {
		// Decode records made by the inline-packing-mode SIV format.
		for trial := 0; trial < trials; trial++ {
			packed, err = openSIV(c.key, c.trialAAD(from, index, sequence, trial), sealed)
			if err == nil {
				break
			}
		}
	}
	if err != nil {
		// Decode records created by the former nonce-bearing AES-GCM format.
		aead, legacyErr := newAEAD(c.key)
		if legacyErr != nil || len(sealed) < aead.NonceSize()+aead.Overhead() {
			return nil, errors.New("chained carrier authentication failed")
		}
		packed, legacyErr = aead.Open(nil, sealed[:aead.NonceSize()], sealed[aead.NonceSize():], c.aad(from, index, sequence))
		if legacyErr != nil {
			return nil, errors.New("chained carrier authentication failed")
		}
	}
	plaintext, err := unpackMessageWithDictionary(packed, c.compressionDictionary())
	if err != nil {
		return nil, fmt.Errorf("unpack chained message: %w", err)
	}
	return plaintext, nil
}

const maxConversationDictionary = 32 << 10

func (c *ConversationChain) compressionDictionary() []byte {
	if len(c.records) == 0 {
		return nil
	}
	capacity := len(chatDictionary)
	for _, record := range c.records {
		capacity += len(record.Encrypted) + 1
	}
	dictionary := make([]byte, 0, capacity)
	dictionary = append(dictionary, chatDictionary...)
	for _, record := range c.records {
		dictionary = append(dictionary, '\n')
		dictionary = append(dictionary, record.Encrypted...)
	}
	if len(dictionary) > maxConversationDictionary {
		dictionary = dictionary[len(dictionary)-maxConversationDictionary:]
	}
	return dictionary
}

func (c *ConversationChain) aad(from string, index, sequence uint64) []byte {
	b := make([]byte, 0, len(c.conversation)+len(from)+96)
	b = append(b, "decalgo-group-aad-v1\x00"...)
	b = append(b, c.conversation...)
	b = append(b, 0)
	b = append(b, from...)
	b = append(b, 0)
	b = append(b, chainNumbers(index, sequence)...)
	b = append(b, c.chain[:]...)
	return b
}

func (c *ConversationChain) trialAAD(from string, index, sequence uint64, trial int) []byte {
	aad := c.aad(from, index, sequence)
	if trial == 0 {
		return aad
	}
	return append(aad, 0, 't', 'r', 'i', 'a', 'l', byte(trial))
}

func (c *ConversationChain) trialModeAAD(from string, index, sequence uint64, trial int, mode byte) []byte {
	aad := c.trialAAD(from, index, sequence, trial)
	return append(aad, 0, 'p', 'a', 'c', 'k', mode)
}

func chainNumbers(index, sequence uint64) []byte {
	b := make([]byte, 16)
	binary.BigEndian.PutUint64(b[:8], index)
	binary.BigEndian.PutUint64(b[8:], sequence)
	return b
}

func validateSender(from string) error {
	if strings.TrimSpace(from) == "" {
		return errors.New("sender name is required")
	}
	if strings.ContainsAny(from, "\r\n\x00") {
		return errors.New("sender name contains a forbidden character")
	}
	if len(from) > 128 {
		return errors.New("sender name is too long")
	}
	return nil
}

// humanWrittenCarrier is deliberately conservative and deterministic. It
// rejects conspicuous generation failures while leaving grammar decisions to
// the language model. Selection among passing trials is then purely by visible
// character count.
func humanWrittenCarrier(text string) bool {
	ok, _ := humanWrittenCarrierReason(text)
	return ok
}

// humanWrittenCarrierReason reports which rule a carrier violated. The gate
// decides whether a generation is good enough to transmit, so when every trial
// fails the caller needs the specific reason to act on. Returning a bare false
// made a total failure indistinguishable from a single bad sample.
func humanWrittenCarrierReason(text string) (bool, string) {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return false, "empty carrier"
	}
	if !endsSentence(trimmed) {
		return false, "carrier does not end on sentence-final punctuation"
	}
	if strings.Count(trimmed, "?") > 4 {
		return false, "carrier asks more than four questions"
	}
	if strings.ContainsAny(trimmed, "\r\n\t") {
		return false, "carrier contains a line break or tab"
	}
	if strings.Contains(trimmed, "  ") {
		return false, "carrier contains a double space"
	}
	words := strings.Fields(strings.ToLower(trimmed))
	if len(words) < 3 {
		return false, fmt.Sprintf("carrier has %d words, fewer than the 3 required", len(words))
	}
	normalized := make([]string, len(words))
	for i, word := range words {
		normalized[i] = strings.Trim(word, ".,!?;:'\"")
		if i > 0 && normalized[i] != "" && normalized[i] == normalized[i-1] {
			return false, fmt.Sprintf("carrier repeats %q back to back", normalized[i])
		}
	}
	seen := make(map[string]struct{}, len(words))
	for i := 0; i+3 < len(normalized); i++ {
		gram := strings.Join(normalized[i:i+4], " ")
		if _, duplicate := seen[gram]; duplicate {
			return false, fmt.Sprintf("carrier repeats the 4-word phrase %q", gram)
		}
		seen[gram] = struct{}{}
	}
	return true, ""
}

func semanticHumanWritten(ctx context.Context, model LanguageModel, text string) (bool, float64, error) {
	prompt := "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" +
		"You are a strict prose reviewer. Decide whether the supplied text could plausibly be one ordinary message written by a real person to a close friend. It must be coherent as a whole, grammatically understandable, stay on one topic, avoid abrupt non sequiturs and excessive questions, and contain no labels, metadata, prompt language, or conspicuous repetition. Minor casual phrasing is fine. Answer only YES or NO." +
		"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nMESSAGE:\nI finally tried that bakery near work, and the cinnamon rolls were incredible. I might go back tomorrow.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nYES" +
		"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nMESSAGE:\nThe movie was funny. Anyway my neighbor owns six lamps. Did you eat? The ending because yesterday.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nNO" +
		"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nMESSAGE:\n" + escapePromptControl(text) +
		"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
	tokens, err := model.Tokenize(ctx, prompt)
	if err != nil {
		return false, 0, fmt.Errorf("tokenize review prompt: %w", err)
	}
	candidates, err := model.Next(ctx, tokens, 256)
	if err != nil {
		return false, 0, fmt.Errorf("score review response: %w", err)
	}
	yesScore, noScore := math.Inf(-1), math.Inf(-1)
	for _, candidate := range candidates {
		answer := strings.Trim(strings.ToLower(strings.TrimSpace(candidate.Text)), ".,!;:'\"")
		switch answer {
		case "yes":
			if candidate.LogProb > yesScore {
				yesScore = candidate.LogProb
			}
		case "no":
			if candidate.LogProb > noScore {
				noScore = candidate.LogProb
			}
		}
	}
	if math.IsInf(yesScore, -1) || math.IsInf(noScore, -1) {
		return false, 0, errors.New("review model did not expose both YES and NO tokens")
	}
	margin := yesScore - noScore
	return margin >= 0, margin, nil
}
